"""Train the fidgeting classifier on extracted MAS pose CSVs.

Group-safe pipeline: windows are never split across the train/validation
boundary; whole SOURCE VIDEO groups are. Group overlap is asserted to be
zero before training proceeds.

Usage:
    python -m symptom_fidgeting.train_fidgeting [--out_dir DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_auc_score, precision_score, recall_score,
                             classification_report)

from . import paths
from .dataset_map import usable_dataset, load_clip_to_source
from .features import window_to_features, feature_names

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS = 120
BATCH_SIZE = 128
VAL_FRACTION = 0.20          # fraction of SOURCE GROUPS held out
EARLY_STOP_PATIENCE = 15
REDUCE_LR_PATIENCE = 8


def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred)
                + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal = tf.pow(1.0 - p_t, gamma) * bce
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * focal)
    loss_fn.__name__ = "focal_loss"
    return loss_fn


def build_model(n_features):
    l2 = tf.keras.regularizers.L2(1e-4)
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,), name="input"),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=l2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
    ], name="fidget_mlp")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=focal_loss(gamma=2.0, alpha=0.75),
              metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                       tf.keras.metrics.AUC(name="auc"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])
    return m


def load_dataset(csv_dir):
    """Load all extracted clip CSVs -> (X, y, clip_ids, groups).

    Windows of WINDOW_FRAMES frames, stepped by WINDOW_STEP, are built per
    clip. Windows whose pose detection rate < threshold are dropped (the
    feature function returns None for them).
    """
    X, y, clips, groups = [], [], [], []
    clip2src = load_clip_to_source()
    pos_clips = neg_clips = 0
    files = sorted(f for f in os.listdir(csv_dir)
                   if f.endswith(".csv") and not f.startswith("."))
    for fname in files:
        df = pd.read_csv(os.path.join(csv_dir, fname))
        if "clip" not in df.columns or "label" not in df.columns:
            continue
        clip = df["clip"].iloc[0]
        group = clip2src.get(clip, df["source_group"].iloc[0])
        label = int(df["label"].iloc[0])
        if label == 1:
            pos_clips += 1
        else:
            neg_clips += 1
        n = len(df)
        for s in range(0, max(0, n - paths.WINDOW_FRAMES) + 1,
                       paths.WINDOW_STEP):
            if s + paths.WINDOW_FRAMES > n:
                break
            feats, _ = window_to_features(
                df.iloc[s:s + paths.WINDOW_FRAMES])
            if feats is None:
                continue
            X.append(feats)
            y.append(label)
            clips.append(clip)
            groups.append(group)
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    return (X, y, np.asarray(clips), np.asarray(groups),
            pos_clips, neg_clips)


class EpochPrinter(tf.keras.callbacks.Callback):
    def __init__(self, total):
        super().__init__()
        self.total = total
        print(f"\n  {'Ep':>3} {'TrLoss':>7} {'TrAcc':>6} "
              f"{'ValLoss':>7} {'ValAcc':>6} {'ValAUC':>6}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"  {epoch+1:>3} {logs.get('loss', 0):>7.4f} "
              f"{logs.get('accuracy', 0):>6.3f} "
              f"{logs.get('val_loss', 0):>7.4f} "
              f"{logs.get('val_accuracy', 0):>6.3f} "
              f"{logs.get('val_auc', 0):>6.3f}")


def find_best_threshold(probs, y, lo=0.20, hi=0.85, step=0.02):
    best_f1, best_thr = -1.0, 0.5
    for thr in np.arange(lo, hi + 1e-9, step):
        preds = (probs > thr).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y.astype(int), preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def evaluate(y, probs, thr, label="val"):
    preds = (probs > thr).astype(int)
    acc = accuracy_score(y.astype(int), preds)
    prec = precision_score(y.astype(int), preds, zero_division=0)
    rec = recall_score(y.astype(int), preds, zero_division=0)
    f1 = f1_score(y.astype(int), preds, zero_division=0)
    auc = roc_auc_score(y.astype(int), probs) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y.astype(int), preds,
                                      labels=[0, 1]).ravel()
    return {"label": label, "threshold": thr, "accuracy": acc,
            "precision_pos": prec, "recall_pos": rec, "f1_pos": f1,
            "roc_auc": auc, "cm_tn": int(tn), "cm_fp": int(fp),
            "cm_fn": int(fn), "cm_tp": int(tp), "n": int(len(y)),
            "n_pos": int(y.sum())}


def print_metrics(m):
    print(f"\n  === {m['label'].upper()} (threshold {m['threshold']:.2f}) ===")
    print(f"  Accuracy  = {m['accuracy']:.3f}")
    print(f"  Precision (positive) = {m['precision_pos']:.3f}")
    print(f"  Recall    (positive) = {m['recall_pos']:.3f}")
    print(f"  F1        (positive) = {m['f1_pos']:.3f}")
    print(f"  ROC AUC             = {m['roc_auc']:.3f}")
    print(f"  n={m['n']} (pos={m['n_pos']})")
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"                    Pred Not-Fidget   Pred Fidget")
    print(f"    Actual Not-Fidget  {m['cm_tn']:>13,}  {m['cm_fp']:>11,}")
    print(f"    Actual Fidget      {m['cm_fn']:>13,}  {m['cm_tp']:>11,}")


def per_group_report(y, probs, groups, thr):
    print("\n  Per-source-group validation results:")
    print(f"  {'group':<46} {'n':>4} {'pos':>3} {'acc':>5} "
          f"{'f1':>5} {'prec':>5} {'rec':>5}")
    gids = sorted(set(groups))
    rows = []
    for g in gids:
        mask = groups == g
        ys = y[mask].astype(int)
        ps = probs[mask]
        if len(ys) < 3:
            continue
        preds = (ps > thr).astype(int)
        rows.append((g, len(ys), ys.mean(),
                     accuracy_score(ys, preds),
                     f1_score(ys, preds, zero_division=0),
                     precision_score(ys, preds, zero_division=0),
                     recall_score(ys, preds, zero_division=0)))
    for g, n, posr, acc, f1, prec, rec in rows:
        print(f"  {g[:46]:<46} {n:>4} {posr*100:>3.0f}% {acc:>5.2f} "
              f"{f1:>5.2f} {prec:>5.2f} {rec:>5.2f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv_dir", default=paths.EXTRACTED_DIR)
    ap.add_argument("--out_dir", default=paths.MODELS_DIR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    out = args.out_dir

    print("=" * 72)
    print("FIDGETING PROTOTYPE — TRAINING")
    print("=" * 72)

    t_load = time.time()
    X, y, clips, groups, pos_c, neg_c = load_dataset(args.csv_dir)
    print(f"Loaded {len(X):,} windows from {pos_c} pos + {neg_c} neg clips "
          f"({time.time()-t_load:.1f}s)")

    n_features = X.shape[1]
    exp_features = feature_names()
    assert n_features == len(exp_features), \
        f"feature count {n_features} != schema {len(exp_features)}"
    print(f"Features per window: {n_features}")

    # --- Leakage-safe group split (by original source video) ---
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION,
                                 random_state=SEED)
    tr_idx, va_idx = next(splitter.split(X, y, groups))
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    g_tr, g_va = groups[tr_idx], groups[va_idx]
    c_tr, c_va = clips[tr_idx], clips[va_idx]

    tr_groups = set(g_tr.tolist())
    va_groups = set(g_va.tolist())
    overlap = tr_groups & va_groups
    print(f"\n  Train: {len(X_tr):,} windows / {len(tr_groups)} groups "
          f"({len(set(c_tr.tolist()))} clips)")
    print(f"  Val:   {len(X_va):,} windows / {len(va_groups)} groups "
          f"({len(set(c_va.tolist()))} clips)")
    print(f"  Group overlap (must be 0): {len(overlap)}")
    if overlap:
        print("  ABORT: source groups leaked across the split!")
        return 1

    # --- Scale on train only ---
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    print(f"  Train pos rate: {y_tr.mean()*100:.1f}%   "
          f"Val pos rate: {y_va.mean()*100:.1f}%")

    model = build_model(n_features)
    model.summary()

    callbacks = [
        EpochPrinter(args.epochs),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=EARLY_STOP_PATIENCE,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                             factor=0.5,
                                             patience=REDUCE_LR_PATIENCE,
                                             min_lr=1e-5),
    ]
    t_tr = time.time()
    hist = model.fit(X_tr_s, y_tr, validation_data=(X_va_s, y_va),
                     epochs=args.epochs, batch_size=BATCH_SIZE,
                     verbose=0, callbacks=callbacks)
    train_seconds = time.time() - t_tr
    best_ep = int(np.argmax(hist.history["val_auc"])) + 1
    print(f"\n  Early-stopped / best epoch {best_ep}; "
          f"training took {train_seconds:.1f}s")

    # --- Final threshold + evaluation (validation) ---
    probs_va = model.predict(X_va_s, verbose=0).flatten()
    probs_tr = model.predict(X_tr_s, verbose=0).flatten()
    thr, _f1 = find_best_threshold(probs_va, y_va)
    val_m = evaluate(y_va, probs_va, thr)
    tr_m = evaluate(y_tr, probs_tr, thr)
    print_metrics(tr_m)
    print_metrics(val_m)
    rows = per_group_report(y_va, probs_va, g_va, thr)

    # --- Save everything needed for inference ---
    model.save(os.path.join(out, "fidget_model.keras"))
    np.save(os.path.join(out, "scaler_mean.npy"), scaler.mean_.astype(np.float32))
    np.save(os.path.join(out, "scaler_scale.npy"), scaler.scale_.astype(np.float32))
    np.save(os.path.join(out, "threshold.npy"), np.array([thr], np.float32))
    np.save(os.path.join(out, "window_frames.npy"),
            np.array([paths.WINDOW_FRAMES], np.int32))
    np.save(os.path.join(out, "window_step.npy"),
            np.array([paths.WINDOW_STEP], np.int32))
    with open(os.path.join(out, "feature_names.json"), "w") as f:
        json.dump(exp_features, f, indent=2)

    metadata = {
        "model": "MLP 64-32-1, focal loss gamma=2 alpha=0.75, L2 1e-4",
        "features": len(exp_features),
        "window_frames": paths.WINDOW_FRAMES,
        "window_step": paths.WINDOW_STEP,
        "min_detection_rate": paths.MIN_DETECTION_RATE,
        "train_windows": int(len(X_tr)), "val_windows": int(len(X_va)),
        "train_groups": len(tr_groups), "val_groups": len(va_groups),
        "group_overlap": len(overlap),
        "train_clips": len(set(c_tr.tolist())),
        "val_clips": len(set(c_va.tolist())),
        "pos_rate_train": float(y_tr.mean()), "pos_rate_val": float(y_va.mean()),
        "threshold": thr,
        "val_metrics": val_m, "train_metrics": tr_m,
        "best_epoch": best_ep, "train_seconds": train_seconds,
        "seed": SEED,
        "label_policy": "pos=contains body stress; neg=['no stress']; "
                        "ignored=other stress-only clips",
        "proxy_statement": "MAS body stress used as proxy for observable "
                           "upper-body fidgeting; NOT a clinical diagnosis.",
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # --- TFLite export ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    tflite_path = os.path.join(out, "fidget_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"\n  Saved: fidget_model.keras, fidget_model.tflite "
          f"({len(tflite_bytes)//1024} KB), scaler_mean/scale.npy, "
          f"threshold.npy, window_*.npy, feature_names.json, metadata.json")
    print("  TFLite exported.")

    summary = {"val": val_m, "train": tr_m, "threshold": thr,
               "val_groups": sorted(va_groups), "per_group": [
                   {"group": g, "n": n, "acc": acc, "f1": f1,
                    "precision": prec, "recall": rec}
                   for g, n, posr, acc, f1, prec, rec in rows]}
    # Persist the held-out clip list (filename/group/label) so the
    # end-to-end inference test can pick unseen validation clips without
    # re-deriving the split.
    clip_meta = {}
    for g, c, lab in zip(g_va.tolist(), c_va.tolist(), y_va.astype(int)):
        clip_meta.setdefault(c, {"group": g, "label": int(lab)})
    with open(os.path.join(out, "val_clips.json"), "w") as f:
        json.dump(clip_meta, f, indent=2)
    with open(os.path.join(paths.EXPERIMENTS_DIR,
                           "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
