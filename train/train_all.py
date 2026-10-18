import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

DATASET_ROOT = Path(r"F:\Thesis\videos\New folder\Thesis Data Gathering")
CSV_PATH = DATASET_ROOT / "symptom_frames.csv"
OUTPUT_DIR = DATASET_ROOT / "symptom_model"

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
EPOCHS = 60
BATCH_SIZE = 64
VAL_SPLIT = 0.20

META_COLUMNS = {
    "angle", "participant_id", "symptom", "video_id", "frame", "timestamp_ms", "fps",
    *LABELS,
}


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


def make_windows(df, feature_columns):
    X, Y, groups = [], [], []
    for _, video in df.groupby("video_id", sort=False):
        video = video.sort_values("frame")
        values = video[feature_columns].to_numpy(dtype=np.float32)
        labels = video[LABELS].to_numpy(dtype=np.float32)
        participant = str(video["participant_id"].iloc[0])

        if len(video) < WINDOW_FRAMES:
            continue

        for start in range(0, len(video) - WINDOW_FRAMES + 1, WINDOW_STEP):
            end = start + WINDOW_FRAMES
            window = values[start:end]
            delta = np.diff(window, axis=0)
            features = np.concatenate([
                np.mean(window, axis=0),
                np.std(window, axis=0),
                np.mean(np.abs(delta), axis=0),
            ])
            X.append(features)
            Y.append((labels[start:end].mean(axis=0) >= 0.5).astype(np.float32))
            groups.append(participant)

    if not X:
        raise ValueError("No training windows were created. Check the CSV and video lengths.")
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(groups)


def best_threshold(probabilities, targets):
    if len(np.unique(targets)) < 2:
        return 0.50
    best_f1, best_value = -1.0, 0.50
    for threshold in np.arange(0.20, 0.801, 0.02):
        predictions = (probabilities >= threshold).astype(np.int32)
        score = f1_score(targets, predictions, zero_division=0)
        if score > best_f1:
            best_f1, best_value = score, float(threshold)
    return best_value


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Training CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    missing = [column for column in [*LABELS, "video_id", "participant_id", "frame"] if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    feature_columns = [
        column for column in df.columns
        if column not in META_COLUMNS
        and (column.startswith(("face_", "bs_", "pose_", "left_hand_", "right_hand_"))
             or column in {"face_detected", "pose_detected", "body_reference_available",
                           "left_hand_detected", "right_hand_detected"})
    ]
    if not feature_columns:
        raise ValueError("No landmark or blendshape features found in the CSV.")

    df[feature_columns] = (
        df[feature_columns].apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    df[LABELS] = df[LABELS].apply(pd.to_numeric, errors="coerce").fillna(0).clip(0, 1)

    X, Y, groups = make_windows(df, feature_columns)
    participants = np.unique(groups)
    if len(participants) < 2:
        raise ValueError("At least two participant IDs are required for participant-level validation.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=42)
    train_idx, val_idx = next(splitter.split(X, Y, groups))
    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(X_train.shape[1])
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=12, restore_best_weights=True, verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=0
            ),
        ],
    )

    probabilities = model.predict(X_val, verbose=0)
    thresholds = np.array([
        best_threshold(probabilities[:, i], Y_val[:, i].astype(np.int32))
        for i in range(len(LABELS))
    ], dtype=np.float32)
    predictions = (probabilities >= thresholds).astype(np.int32)

    print(f"Participants: {len(participants)} | Videos: {df['video_id'].nunique()} | Windows: {len(X)}")
    print(f"Training windows: {len(train_idx)} | Validation windows: {len(val_idx)}")
    print("Validation metrics:")
    for i, label in enumerate(LABELS):
        y_true = Y_val[:, i].astype(np.int32)
        y_pred = predictions[:, i]
        print(
            f"{label.removeprefix('label_')}: F1={f1_score(y_true, y_pred, zero_division=0):.3f} "
            f"Precision={precision_score(y_true, y_pred, zero_division=0):.3f} "
            f"Recall={recall_score(y_true, y_pred, zero_division=0):.3f} "
            f"Accuracy={accuracy_score(y_true, y_pred):.3f} "
            f"Threshold={thresholds[i]:.2f}"
        )

    model.save(OUTPUT_DIR / "symptom_model.keras")
    np.save(OUTPUT_DIR / "scaler_mean.npy", scaler.mean_.astype(np.float32))
    np.save(OUTPUT_DIR / "scaler_scale.npy", scaler.scale_.astype(np.float32))
    np.save(OUTPUT_DIR / "thresholds.npy", thresholds)
    with (OUTPUT_DIR / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file)
    with (OUTPUT_DIR / "window_feature_order.json").open("w", encoding="utf-8") as file:
        json.dump(
            [f"{stat}_{column}" for stat in ("mean", "std", "mean_abs_delta") for column in feature_columns],
            file,
        )
    with (OUTPUT_DIR / "label_columns.json").open("w", encoding="utf-8") as file:
        json.dump(LABELS, file)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    (OUTPUT_DIR / "symptom_model.tflite").write_bytes(tflite_model)

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"Best epoch: {best_epoch}/{len(history.history['loss'])}")
    print(f"Model: {OUTPUT_DIR / 'symptom_model.tflite'}")


if __name__ == "__main__":
    main()
