import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(42)
tf.random.set_seed(42)

# Paths
LANDMARKS_DIR = r"F:\Thesis\videos\data\landmarks_cremad_fear"
OUTPUT_DIR    = r"F:\Thesis\output\fear"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_FRAMES = 30
WINDOW_STEP   = 5

USE_CREMA = True

EPOCHS      = 60
BATCH_SIZE  = 64
VAL_SPLIT   = 0.20

# FOCAL LOSS

def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce     = (-y_true * tf.math.log(y_pred)
                   - (1 - y_true) * tf.math.log(1 - y_pred))
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal   = tf.pow(1.0 - p_t, gamma) * bce
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * focal)
    loss_fn.__name__ = "focal_loss"
    return loss_fn

# FEATURE EXTRACTION

def stat(s):
    a = np.asarray(s, np.float32)
    return (0.0, 0.0) if len(a) == 0 else (float(np.mean(a)), float(np.std(a)))


def extract_window_features(df):
    def get(col, default=0.0):
        return df[col].fillna(default) if col in df.columns \
               else pd.Series([default] * len(df))
 
    face_detected = get("face_detected", 1.0)
    det_rate      = float(face_detected.mean())
    if det_rate < 0.60:
        return None, None
 
    det_mask = face_detected.values > 0.5
    df_det   = df[det_mask]
 
    def gd(col, default=0.0):
        return df_det[col].fillna(default) if col in df_det.columns \
               else pd.Series([default] * len(df_det))
 
    if len(df_det) >= 3:
        # Eyelid opening: lower-lid landmark minus upper-lid landmark.
        left_eye_open  = gd("lm145_y") - gd("lm159_y")
        right_eye_open = gd("lm374_y") - gd("lm386_y")
        # Mouth opening/width using standard lip-center and corner landmarks.
        mouth_open  = gd("lm14_y") - gd("lm13_y")
        mouth_width = (gd("lm291_x") - gd("lm61_x")).abs()
        # Brow-to-eye distance (same landmark pairs as the old tension geometry).
        brow_eye_l = gd("lm159_y") - gd("lm55_y")
        brow_eye_r = gd("lm386_y") - gd("lm285_y")
 
        eye_l_m, eye_l_s     = stat(left_eye_open)
        eye_r_m, eye_r_s     = stat(right_eye_open)
        mouth_o_m, mouth_o_s = stat(mouth_open)
        mouth_w_m, mouth_w_s = stat(mouth_width)
        brow_l_m, brow_l_s   = stat(brow_eye_l)
        brow_r_m, brow_r_s   = stat(brow_eye_r)
    else:
        eye_l_m = eye_l_s = eye_r_m = eye_r_s = 0.0
        mouth_o_m = mouth_o_s = mouth_w_m = mouth_w_s = 0.0
        brow_l_m = brow_l_s = brow_r_m = brow_r_s = 0.0
 
    biu  = get("bs_browInnerUp")
    boul = get("bs_browOuterUpLeft"); bour = get("bs_browOuterUpRight")
    ewl  = get("bs_eyeWideLeft");     ewr  = get("bs_eyeWideRight")
    jaw  = get("bs_jawOpen")
    msl  = get("bs_mouthStretchLeft"); msr = get("bs_mouthStretchRight")
 
    biu_m, biu_s   = stat(biu)
    boul_m, boul_s = stat(boul)
    bour_m, bour_s = stat(bour)
    ewl_m, ewl_s   = stat(ewl)
    ewr_m, ewr_s   = stat(ewr)
    jaw_m, jaw_s   = stat(jaw)
    msl_m, msl_s   = stat(msl)
    msr_m, msr_s   = stat(msr)
 
    feats = [
        biu_m, biu_s, boul_m, boul_s, bour_m, bour_s,
        ewl_m, ewl_s, ewr_m, ewr_s,
        jaw_m, jaw_s, msl_m, msl_s, msr_m, msr_s,
        eye_l_m, eye_l_s, eye_r_m, eye_r_s,
        mouth_o_m, mouth_o_s, mouth_w_m, mouth_w_s,
        brow_l_m, brow_l_s, brow_r_m, brow_r_s,
        det_rate,
    ]
    return feats, det_rate
 

# def extract_single_image_features(row, rng):
#     def g(col):
#         if isinstance(row, dict): return float(row.get(col, 0.0))
#         try: return float(row[col])
#         except: return 0.0

#     if g("face_detected") < 0.5:
#         return None

#     bg_l  = g("lm159_y") - g("lm55_y")
#     bg_r  = g("lm386_y") - g("lm285_y")
#     bg_avg = (bg_l + bg_r) / 2.0
#     ibd   = abs(g("lm285_x") - g("lm55_x"))
#     bdl = g("bs_browDownLeft"); bdr = g("bs_browDownRight")
#     biu = g("bs_browInnerUp")
#     esl = g("bs_eyeSquintLeft"); esr = g("bs_eyeSquintRight")
#     jaw = g("bs_jawOpen")
#     mpl = g("bs_mouthPressLeft"); mpr = g("bs_mouthPressRight")
#     bt   = (bdl + bdr) / 2.0 - 0.5 * biu
#     es_avg = (esl + esr) / 2.0
#     comp = (esl + esr + mpl + mpr + (1.0 - jaw) + bt) / 6.0
#     ns = lambda: float(abs(rng.normal(0, IMAGE_STD_NOISE)))
#     return [
#         bg_l, ns(), bg_r, ns(), bg_avg, ns(), ibd, ns(),
#         bdl, ns(), bdr, ns(), biu, ns(), bt, ns(),
#         esl, ns(), esr, ns(), es_avg,
#         jaw, ns(), mpl, mpr, comp, ns(), 1.0,
#     ]


# LOAD DATASET

def load_dataset():
    X_all, y_all, groups_all = [], [], []
 
    n_fear_videos = 0
    n_neutral_videos = 0
    total_frames = 0
    n_windows_fear = 0
    n_windows_neutral = 0
    actors = set()
 
    if USE_CREMA:
        csv_files = sorted(
            f for f in glob.glob(os.path.join(LANDMARKS_DIR, "*.csv"))
            if not os.path.basename(f).endswith("_baseline.csv")
        )
 
        for csv_path in csv_files:
            df = pd.read_csv(csv_path).fillna(0.0)
            if "actor" not in df.columns or "label" not in df.columns:
                continue
 
            actor = str(df["actor"].iloc[0])
            label = int(df["label"].iloc[0])  # FEA -> 1, NEU -> 0
 
            actors.add(actor)
            total_frames += len(df)
            if label == 1:
                n_fear_videos += 1
            else:
                n_neutral_videos += 1
 
            n = len(df)
            for s in range(0, n - WINDOW_FRAMES, WINDOW_STEP):
                feats, _ = extract_window_features(df.iloc[s:s + WINDOW_FRAMES])
                if feats is None:
                    continue
                X_all.append(np.array(feats, np.float32))
                y_all.append(label)
                groups_all.append(actor)
                if label == 1:
                    n_windows_fear += 1
                else:
                    n_windows_neutral += 1
    else:
        csv_files = []

    print(f"  Videos loaded:   {len(csv_files):,}")
    print(f"  Fear videos:     {n_fear_videos:,}")
    print(f"  Neutral videos:  {n_neutral_videos:,}")
    print(f"  Actors:          {len(actors):,}")
    print(f"  Total frames:    {total_frames:,}")
    print(f"  Total windows:   {len(X_all):,}")
    print(f"  Fear windows:    {n_windows_fear:,}")
    print(f"  Neutral windows: {n_windows_neutral:,}")
 
    return (np.array(X_all, np.float32),
            np.array(y_all, np.float32),
            np.array(groups_all))

# MODEL

def build_model():
    l2 = tf.keras.regularizers.L2(0.005)
    m  = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,), name="input"),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.40),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(1, activation='sigmoid', name="output"),
    ], name="fear")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    return m


class EpochPrinter(tf.keras.callbacks.Callback):

    def __init__(self, total_epochs):
        super().__init__()
        self.total = total_epochs
        print(f"\n  {'Epoch':>5}  {'Train Loss':>10}  "
              f"{'Train Acc':>9}  {'Val Loss':>8}  "
              f"{'Val Acc':>7}  {'Val AUC':>7}")
        print(f"  {'-'*60}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"  {epoch+1:>5}  "
              f"{logs.get('loss', 0):>10.4f}  "
              f"{logs.get('accuracy', 0):>9.4f}  "
              f"{logs.get('val_loss', 0):>8.4f}  "
              f"{logs.get('val_accuracy', 0):>7.4f}  "
              f"{logs.get('val_auc', 0):>7.4f}")


# TRAIN AND EVALUATE

def find_best_threshold(probs, y):
    best_f1, best_thr = 0.0, 0.50
    for thr in np.arange(0.20, 0.80, 0.02):
        preds = (probs > thr).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y.astype(int), preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def evaluate_and_print(model, scaler, X_val, y_val, groups_val):
    probs = model.predict(scaler.transform(X_val), verbose=0).flatten()
    opt_thr, opt_f1 = find_best_threshold(probs, y_val)
    preds = (probs > opt_thr).astype(int)

    acc  = accuracy_score(y_val.astype(int), preds)
    rep  = classification_report(y_val.astype(int), preds,
                                  output_dict=True, zero_division=0)
    prec = rep.get('1', {}).get('precision', 0.0)
    rec  = rep.get('1', {}).get('recall', 0.0)

    print(f"\n  Evaluation at optimal threshold {opt_thr:.2f}:")
    print(f"  F1      = {opt_f1:.3f}")
    print(f"  Accuracy = {acc:.3f}")
    print(f"  Precision = {prec:.3f}")
    print(f"  Recall   = {rec:.3f}")

    cm = confusion_matrix(y_val.astype(int), preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  Confusion matrix:")
        print(f"                    Predicted Neutral   Predicted Fear")
        print(f"    Actual Neutral  {tn:>18,}  {fp:>15,}")
        print(f"    Actual Fear    {fn:>18,}  {tp:>15,}")

    # Per-subject breakdown in val set
    val_subjects = np.unique(groups_val)
    val_subjects = [s for s in val_subjects
                    if s not in ("FERPlus", "AffectNet", "RAVDESS")]
    if val_subjects:
        print(f"\n  Per-subject breakdown (validation DISFA subjects):")
        print(f"  {'Subject':<8} {'Pos%':>5} {'F1':>6} "
              f"{'Acc':>6} {'Prec':>6} {'Rec':>6}")
        print(f"  {'-'*42}")
        for sid in sorted(val_subjects):
            mask = groups_val == sid
            if mask.sum() < 10:
                continue
            p_s = probs[mask]
            y_s = y_val[mask]
            t_s, _ = find_best_threshold(p_s, y_s)
            pr_s = (p_s > t_s).astype(int)
            rs = classification_report(y_s.astype(int), pr_s,
                                       output_dict=True, zero_division=0)
            f1_s  = rs.get('1', {}).get('f1-score', 0.0)
            acc_s = accuracy_score(y_s.astype(int), pr_s)
            pre_s = rs.get('1', {}).get('precision', 0.0)
            rec_s = rs.get('1', {}).get('recall', 0.0)
            print(f"  {sid:<8} {y_s.mean()*100:>4.0f}%  "
                  f"{f1_s:>6.3f} {acc_s:>6.3f} "
                  f"{pre_s:>6.3f} {rec_s:>6.3f}")

    return opt_thr, opt_f1, acc


# TFLITE

def convert_tflite(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    with open(path, 'wb') as f:
        f.write(tflite_bytes)
    print(f"  TFLite saved: {path} ({len(tflite_bytes)//1024} KB)")



# MAIN

if __name__ == "__main__":
    enabled = []

    print("="*65)
    print("FEAR TRAINING")
    print(f"Datasets: CREMA / FEAR AND NEUTRAL")
    print(f"Val split: {VAL_SPLIT*100:.0f}% of subjects held out")
    print(f"Epochs: {EPOCHS}  Batch size: {BATCH_SIZE}")
    print("="*65)

    print("\nLoading all datasets")
    X, y, groups = load_dataset()

    if len(X) == 0:
        print("No data loaded. Check paths.")
        exit(1)

    n_features = X.shape[1]
    print(f"\nFeature vector length: {n_features}")
    print(f"Total windows: {len(X):,},  pos={y.mean()*100:.1f}%")

    # Group split: entire subjects to train or val, prevents data leakage
    
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SPLIT, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    g_val          = groups[val_idx]

    # Report which subjects went to val
    val_disfa_subjects = [s for s in np.unique(g_val)
                          if s not in ("FERPlus", "AffectNet", "RAVDESS")]
    print(f"\n  Validation subjects: {sorted(val_disfa_subjects)}")
    print(f"  Train windows: {len(X_train):,}  "
          f"pos={y_train.mean()*100:.1f}%")
    print(f"  Val windows:   {len(X_val):,}  "
          f"pos={y_val.mean()*100:.1f}%")

    # Scale
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_val)

    model = build_model()
    model.summary()

    # Epoch-visible training
    print(f"\nTraining with epoch-by-epoch output:")
    history = model.fit(
        X_tr_sc, y_train,
        validation_data=(X_va_sc, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[
            EpochPrinter(EPOCHS),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', mode='max',
                patience=12, restore_best_weights=True,
                verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', mode='max',
                factor=0.5, patience=6,
                min_lr=1e-5, verbose=1),
        ],
    )

    # Best epoch summary
    best_ep  = int(np.argmax(history.history['val_auc'])) + 1
    best_auc = max(history.history['val_auc'])
    best_acc = history.history['val_accuracy'][best_ep - 1]
    print(f"\n  Best epoch: {best_ep}  "
          f"val_auc={best_auc:.4f}  "
          f"val_accuracy={best_acc:.4f}")

    # Full evaluation
    print("\nEvaluating on validation set")
    thr, f1, acc = evaluate_and_print(model, scaler, X_val, y_val, g_val)

    # Save
    np.save(f"{OUTPUT_DIR}/scaler_mean_b.npy",
            scaler.mean_.astype(np.float32))
    np.save(f"{OUTPUT_DIR}/scaler_std_b.npy",
            scaler.scale_.astype(np.float32))
    np.save(f"{OUTPUT_DIR}/threshold_b.npy", np.array([thr]))

    print("\nConverting to TFLite")
    tflite_path = f"{OUTPUT_DIR}/fear_model.tflite"
    convert_tflite(model, tflite_path)

    print(f"\n{'='*65}")
    print(f"DONE")
    print(f"  F1 (opt thr={thr:.2f}): {f1:.3f}")
    print(f"  Accuracy:              {acc:.3f}")
    print(f"  Best epoch:            {best_ep}/{EPOCHS}")
    print(f"  Val AUC:               {best_auc:.4f}")
