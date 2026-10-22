import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import gc
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

SEED = 42
DATASET_ROOT = Path(r"F:\Thesis\videos\New folder\Thesis Data Gathering")
CSV_PATH = DATASET_ROOT / "symptom_frames.csv.gz"
if not CSV_PATH.exists():
    CSV_PATH = DATASET_ROOT / "symptom_frames.csv"
OUTPUT_DIR = DATASET_ROOT / "symptom_model_training_trial"

LABELS = [
    "label_facial_tension",
    "label_hand_tremor",
    "label_heavy_breathing",
    "label_knee_tremor",
    "label_fidgeting",
    "label_avoidant_gaze",
    "label_fear",
]

WINDOW_FRAMES = 30
WINDOW_STEP = 5
CHUNK_ROWS = 5000
SCALER_BATCH = 512
EPOCHS = 60
BATCH_SIZE = 64
VAL_SPLIT = 0.20
TEST_SPLIT = 0.20

META_COLUMNS = {
    "angle", "participant_id", "symptom", "video_id", "frame", "timestamp_ms", "fps",
    *LABELS, "label_control", "folder_label_control", "rule_reason",
}
QUALITY_COLUMNS = {
    "face_detected", "pose_detected", "body_reference_available",
    "left_hand_detected", "right_hand_detected",
}

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    loss_fn.__name__ = "focal_loss"
    return loss_fn


def build_model(n_features):
    regularizer = tf.keras.regularizers.L2(0.005)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,), name="input"),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.40),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(len(LABELS), activation="sigmoid", name="symptoms"),
    ], name="anxiety_symptoms")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=focal_loss(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=len(LABELS)),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_feature_columns(columns):
    prefixes = ("face_", "face_matrix_", "bs_", "pose_", "left_hand_", "right_hand_")
    selected = [column for column in columns if column.startswith(prefixes)]
    selected.extend(column for column in QUALITY_COLUMNS if column in columns)
    selected = [column for column in selected if column not in META_COLUMNS]
    return list(dict.fromkeys(selected))


def iter_videos(usecols):
    pending = None
    for chunk in pd.read_csv(
        CSV_PATH, usecols=usecols, chunksize=CHUNK_ROWS, compression="infer", low_memory=False
    ):
        combined = chunk if pending is None else pd.concat([pending, chunk], ignore_index=True)
        if combined.empty:
            continue
        ids = combined["video_id"].astype(str).to_numpy()
        last_id = ids[-1]
        different = np.flatnonzero(ids != last_id)
        if len(different) == 0:
            pending = combined
            continue
        split_at = int(different[-1] + 1)
        complete = combined.iloc[:split_at]
        pending = combined.iloc[split_at:].copy()
        for _, video in complete.groupby("video_id", sort=False):
            yield video
    if pending is not None and not pending.empty:
        for _, video in pending.groupby("video_id", sort=False):
            yield video


def make_windows(video, feature_columns, x_store, y_store, groups, window_meta, cursor, rule_labels):
    video = video.sort_values("frame")
    if len(video) < WINDOW_FRAMES:
        return cursor

    values = (
        video[feature_columns].apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    )
    labels = video[LABELS].apply(pd.to_numeric, errors="coerce").fillna(0).clip(0, 1).to_numpy(dtype=np.float32)

    feature_windows = sliding_window_view(values, WINDOW_FRAMES, axis=0)[::WINDOW_STEP]
    delta_windows = np.diff(feature_windows, axis=2)
    features = np.concatenate([
        feature_windows.mean(axis=2),
        feature_windows.std(axis=2),
        np.abs(delta_windows).mean(axis=2),
    ], axis=1).astype(np.float32, copy=False)

    label_windows = sliding_window_view(labels, WINDOW_FRAMES, axis=0)[::WINDOW_STEP]
    targets = (label_windows.mean(axis=2) >= 0.5).astype(np.float32)

    if rule_labels:
        control = pd.to_numeric(video["label_control"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        any_labeled = (labels.max(axis=1) > 0.5) | (control > 0.5)
        coverage_windows = sliding_window_view(any_labeled.astype(np.float32), WINDOW_FRAMES)[::WINDOW_STEP]
        keep = coverage_windows.mean(axis=1) >= 0.50
        features = features[keep]
        targets = targets[keep]
        starts = np.arange(0, len(video) - WINDOW_FRAMES + 1, WINDOW_STEP)[keep]
    else:
        starts = np.arange(0, len(video) - WINDOW_FRAMES + 1, WINDOW_STEP)

    n = len(features)
    if n == 0:
        return cursor

    end = cursor + n
    x_store[cursor:end] = features
    y_store[cursor:end] = targets

    participant = str(video["participant_id"].iloc[0])
    angle = str(video["angle"].iloc[0]) if "angle" in video.columns else "unknown"
    symptom = str(video["symptom"].iloc[0]) if "symptom" in video.columns else "unknown"
    video_id = str(video["video_id"].iloc[0])
    fps = float(pd.to_numeric(video.get("fps", pd.Series([30.0])), errors="coerce").fillna(30.0).iloc[0])
    for start in starts:
        groups[cursor] = participant
        window_meta.append({
            "video_id": video_id,
            "participant_id": participant,
            "angle": angle,
            "folder_symptom": symptom,
            "start_frame": int(video["frame"].iloc[start]),
            "end_frame": int(video["frame"].iloc[start + WINDOW_FRAMES - 1]),
            "start_time_seconds": float(video["timestamp_ms"].iloc[start] / 1000.0) if "timestamp_ms" in video.columns else float(start / max(fps, 1.0)),
        })
        cursor += 1
    return cursor


def best_threshold(probabilities, targets):
    if len(np.unique(targets)) < 2:
        return 0.50
    best_f1, best_value = -1.0, 0.50
    for threshold in np.arange(0.10, 0.901, 0.02):
        predictions = (probabilities >= threshold).astype(np.int32)
        score = f1_score(targets, predictions, zero_division=0)
        if score > best_f1:
            best_f1, best_value = score, float(threshold)
    return best_value


def make_dataset(x_store, y_store, indices, n_features, shuffle=False):
    indices = np.asarray(indices, dtype=np.int64)

    def generator():
        order = indices.copy()
        if shuffle:
            np.random.shuffle(order)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            yield np.asarray(x_store[batch], dtype=np.float32), np.asarray(y_store[batch], dtype=np.float32)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(LABELS)), dtype=tf.float32),
        ),
    )
    return dataset.prefetch(1)


def predict_indices(model, x_store, indices):
    output = np.zeros((len(indices), len(LABELS)), dtype=np.float32)
    for start in range(0, len(indices), BATCH_SIZE * 4):
        batch_indices = indices[start:start + BATCH_SIZE * 4]
        batch = np.asarray(x_store[batch_indices], dtype=np.float32)
        output[start:start + len(batch)] = model.predict(batch, batch_size=BATCH_SIZE, verbose=0)
    return output


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Training CSV not found: {CSV_PATH}")

    header = pd.read_csv(CSV_PATH, compression="infer", nrows=0).columns.tolist()
    required = [*LABELS, "video_id", "participant_id", "frame"]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    feature_columns = get_feature_columns(header)
    if not feature_columns:
        raise ValueError("No landmark or blendshape features found in the CSV.")

    rule_labels = "label_control" in header and "rule_reason" in header
    if rule_labels:
        print("Label source: rule-reclassified labels; no_rule_match windows are excluded.")
    else:
        print("Label source: original folder labels.")

    usecols = list(dict.fromkeys([
        *feature_columns, *required,
        *[column for column in ("angle", "symptom", "timestamp_ms", "fps") if column in header],
        *( ["label_control", "rule_reason"] if rule_labels else [] ),
    ]))
    meta = pd.read_csv(CSV_PATH, usecols=["video_id", "participant_id"], compression="infer", low_memory=False)
    counts = meta.groupby("video_id", sort=False).size().to_numpy(dtype=np.int64)
    max_windows = int(sum(max(0, (int(count) - WINDOW_FRAMES) // WINDOW_STEP + 1) for count in counts))
    participant_ids = sorted(meta["participant_id"].astype(str).unique())
    del meta
    if max_windows == 0:
        raise ValueError("No windows could be made from the source videos.")

    n_features = len(feature_columns) * 3
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="training_cache_", dir=OUTPUT_DIR))
    x_path = temp_dir / "window_features.dat"
    y_path = temp_dir / "window_labels.dat"
    x_store = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(max_windows, n_features))
    y_store = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(max_windows, len(LABELS)))
    groups = np.empty(max_windows, dtype="U64")
    window_meta = []
    cursor = 0

    try:
        for video in iter_videos(usecols):
            cursor = make_windows(video, feature_columns, x_store, y_store, groups, window_meta, cursor, rule_labels)
        if cursor == 0:
            raise ValueError("No eligible training windows were created.")
        x_store.flush()
        y_store.flush()

        X = x_store[:cursor]
        Y = y_store[:cursor]
        groups = groups[:cursor]
        window_meta = window_meta[:cursor]
        unique_groups = np.unique(groups)
        if len(unique_groups) < 3:
            raise ValueError("At least three participant IDs are required for train/validation/test splits.")

        first_split = GroupShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=SEED)
        train_val_idx, test_idx = next(first_split.split(np.zeros(cursor), Y, groups))
        val_fraction = VAL_SPLIT / (1.0 - TEST_SPLIT)
        second_split = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=SEED + 1)
        train_rel, val_rel = next(second_split.split(np.zeros(len(train_val_idx)), Y[train_val_idx], groups[train_val_idx]))
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]

        scaler = StandardScaler()
        for start in range(0, len(train_idx), SCALER_BATCH):
            scaler.partial_fit(np.asarray(X[train_idx[start:start + SCALER_BATCH]], dtype=np.float32))
        for start in range(0, cursor, SCALER_BATCH):
            end = min(start + SCALER_BATCH, cursor)
            X[start:end] = scaler.transform(np.asarray(X[start:end], dtype=np.float32)).astype(np.float32)
        x_store.flush()

        model = build_model(n_features)
        train_ds = make_dataset(X, Y, train_idx, n_features, shuffle=True)
        val_ds = make_dataset(X, Y, val_idx, n_features)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=math.ceil(len(train_idx) / BATCH_SIZE),
            validation_steps=math.ceil(len(val_idx) / BATCH_SIZE),
            verbose=2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=0),
            ],
        )

        val_probabilities = predict_indices(model, X, val_idx)
        thresholds = np.array([
            best_threshold(val_probabilities[:, i], Y[val_idx, i].astype(np.int32))
            for i in range(len(LABELS))
        ], dtype=np.float32)
        test_probabilities = predict_indices(model, X, test_idx)
        test_predictions = (test_probabilities >= thresholds).astype(np.int32)

        print(f"Participants: {len(unique_groups)} | Videos: {len(counts)} | Windows: {cursor}")
        print(f"Training/validation/test windows: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
        print(f"Test macro F1: {f1_score(Y[test_idx].astype(np.int32), test_predictions, average='macro', zero_division=0):.3f} | "
              f"Micro F1: {f1_score(Y[test_idx].astype(np.int32), test_predictions, average='micro', zero_division=0):.3f} | "
              f"Exact-match accuracy: {accuracy_score(Y[test_idx].astype(np.int32), test_predictions):.3f}")
        print("Test metrics by symptom:")
        for i, label in enumerate(LABELS):
            truth = Y[test_idx, i].astype(np.int32)
            pred = test_predictions[:, i]
            print(
                f"{label.removeprefix('label_')}: F1={f1_score(truth, pred, zero_division=0):.3f} "
                f"Precision={precision_score(truth, pred, zero_division=0):.3f} "
                f"Recall={recall_score(truth, pred, zero_division=0):.3f} "
                f"Accuracy={accuracy_score(truth, pred):.3f} Threshold={thresholds[i]:.2f}"
            )

        model.save(OUTPUT_DIR / "symptom_model.keras")
        np.save(OUTPUT_DIR / "scaler_mean.npy", scaler.mean_.astype(np.float32))
        np.save(OUTPUT_DIR / "scaler_scale.npy", scaler.scale_.astype(np.float32))
        np.save(OUTPUT_DIR / "thresholds.npy", thresholds)
        with (OUTPUT_DIR / "feature_columns.json").open("w", encoding="utf-8") as file:
            json.dump(feature_columns, file)
        with (OUTPUT_DIR / "window_feature_order.json").open("w", encoding="utf-8") as file:
            json.dump([f"{stat}_{column}" for stat in ("mean", "std", "mean_abs_delta") for column in feature_columns], file)
        with (OUTPUT_DIR / "label_columns.json").open("w", encoding="utf-8") as file:
            json.dump(LABELS, file)
        with (OUTPUT_DIR / "training_source.json").open("w", encoding="utf-8") as file:
            json.dump({"csv_path": str(CSV_PATH), "rule_labels": rule_labels, "participants": unique_groups.tolist()}, file, indent=2)

        prediction_rows = []
        test_meta = pd.DataFrame([window_meta[i] for i in test_idx])
        test_meta["true_labels"] = [";".join(label.removeprefix("label_") for j, label in enumerate(LABELS) if Y[idx, j] >= 0.5) or "none" for idx in test_idx]
        test_meta["predicted_labels"] = [";".join(label.removeprefix("label_") for j, label in enumerate(LABELS) if test_predictions[row, j]) or "none" for row, idx in enumerate(test_idx)]
        for i, label in enumerate(LABELS):
            test_meta[f"true_{label}"] = Y[test_idx, i].astype(np.int8)
            test_meta[f"prob_{label.removeprefix('label_')}"] = test_probabilities[:, i]
            test_meta[f"pred_{label.removeprefix('label_')}"] = test_predictions[:, i]
        test_meta.to_csv(OUTPUT_DIR / "test_window_predictions.csv", index=False)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        (OUTPUT_DIR / "symptom_model.tflite").write_bytes(converter.convert())
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        print(f"Best epoch: {best_epoch}/{len(history.history['loss'])}")
        print(f"Model: {OUTPUT_DIR / 'symptom_model.tflite'}")
        print(f"Test predictions: {OUTPUT_DIR / 'test_window_predictions.csv'}")

        del train_ds, val_ds, model, X, Y
        x_store.flush()
        y_store.flush()
    finally:
        try:
            del train_ds, val_ds
        except UnboundLocalError:
            pass
        try:
            del model, X, Y
        except UnboundLocalError:
            pass
        try:
            x_store.flush()
            y_store.flush()
            x_store._mmap.close()
            y_store._mmap.close()
        except Exception:
            pass
        del x_store, y_store
        gc.collect()
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            print(f"Temporary cache could not be removed; close Python and delete: {temp_dir}")


if __name__ == "__main__":
    main()
