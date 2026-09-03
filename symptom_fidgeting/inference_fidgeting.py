"""Inference for the fidgeting prototype — production path.

Loads the SAVED artifacts (keras/tflite model + scaler + feature schema +
threshold) and runs video -> MediaPipe Pose -> windows -> features ->
fidget probability, using EXACTLY the same feature code as training.

Usage:
    python -m symptom_fidgeting.inference_fidgeting <video> [more videos...]
    python -m symptom_fidgeting.inference_fidgeting --heldout (default demo)
"""
import argparse
import json
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from . import paths
from .features import window_to_features, feature_names
from .pose_utils import (make_pose_landmarker, landmark_dict, DETECTED_COL,
                         _col)
from .dataset_map import load_clip_to_source


class FidgetDetector:
    """Reloadable detector built purely from saved artifacts on disk."""

    def __init__(self, model_dir=paths.MODELS_DIR, use_tflite=None):
        self.dir = model_dir
        # scaler
        self.mean = np.load(os.path.join(model_dir, "scaler_mean.npy"))
        self.scale = np.load(os.path.join(model_dir, "scaler_scale.npy"))
        self.threshold = float(np.load(os.path.join(model_dir,
                                                    "threshold.npy"))[0])
        self.window = int(np.load(os.path.join(model_dir,
                                               "window_frames.npy"))[0])
        try:
            self.step = int(np.load(os.path.join(model_dir,
                                                 "window_step.npy"))[0])
        except Exception:
            self.step = paths.WINDOW_STEP
        with open(os.path.join(model_dir, "feature_names.json")) as f:
            self.features = json.load(f)
        # model
        self.use_tflite = (use_tflite if use_tflite is not None
                           else os.path.exists(os.path.join(
                               model_dir, "fidget_model.tflite")))
        if self.use_tflite:
            import tensorflow as tf
            self.interp = tf.lite.Interpreter(
                model_path=os.path.join(model_dir, "fidget_model.tflite"))
            self.interp.allocate_tensors()
            self.in_ = self.interp.get_input_details()[0]
            self.out = self.interp.get_output_details()[0]
            self.model = None
        else:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(
                os.path.join(model_dir, "fidget_model.keras"),
                custom_objects={"focal_loss": focal_loss_for_reload()})
            self.interp = None
        self.landmarker = None

    def _ensure_landmarker(self):
        if self.landmarker is None:
            self.landmarker = make_pose_landmarker()
        return self.landmarker

    def predict_windows(self, frames_rgb, fps):
        """frames_rgb: list of RGB arrays (or generator). Returns list of
        (window_index, probability) for every full 30-frame window."""
        lm = self._ensure_landmarker()
        rows = []
        for i, rgb in enumerate(frames_rgb):
            ts_ms = int(round(i * 1000.0 / fps))
            if rgb.shape[1] > 640:
                rgb = cv2.resize(rgb, (640, 360))
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = lm.detect_for_video(mp_img, ts_ms)
            row = {DETECTED_COL: 1.0 if res.pose_landmarks else 0.0}
            if res.pose_landmarks:
                row.update(landmark_dict(res.pose_landmarks[0]))
            rows.append(row)
        df = pd.DataFrame(rows)
        out = []
        for s in range(0, max(0, len(df) - self.window) + 1, self.step):
            if s + self.window > len(df):
                break
            feats, _ = window_to_features(df.iloc[s:s + self.window])
            if feats is None:
                continue
            x = (feats - self.mean) / self.scale
            x = x.astype(np.float32).reshape(1, -1)
            if self.interp is not None:
                self.interp.set_tensor(self.in_["index"], x)
                self.interp.invoke()
                prob = float(self.interp.get_tensor(self.out["index"])[0][0])
            else:
                prob = float(self.model.predict(x, verbose=0)[0][0])
            out.append((s, prob))
        return out

    def predict_video(self, video_path, verbose=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        # MediaPipe VIDEO mode requires strictly increasing timestamps across
        # the whole landmarker lifetime; reset per video so ts restarts at 0.
        self.landmarker = None
        t0 = time.time()
        windows = self.predict_windows(frames, fps)
        dt = time.time() - t0
        if not windows:
            return None, 0.0, 0.0, 0.0, fps
        probs = np.array([p for _, p in windows])
        prob = float(np.mean(probs))  # clip-level = mean over windows
        pred = int(prob > self.threshold)
        return windows, prob, pred, dt, fps


def focal_loss_for_reload():
    """Return the exact focal loss used at training (needed to reload
    .keras without retraining)."""
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred)
                + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal = tf.pow(1.0 - p_t, 2.0) * bce
        alpha_t = y_true * 0.75 + (1 - y_true) * 0.25
        return tf.reduce_mean(alpha_t * focal)
    return loss_fn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", help="video paths to classify")
    ap.add_argument("--model_dir", default=paths.MODELS_DIR)
    ap.add_argument("--no-tflite", action="store_true",
                    help="use the .keras model instead of TFLite")
    args = ap.parse_args(argv)

    det = FidgetDetector(args.model_dir, use_tflite=not args.no_tflite)
    print(f"Loaded detector: tflite={det.use_tflite} window={det.window} "
          f"threshold={det.threshold:.3f} features={len(det.features)}")

    videos = args.videos
    if not videos:
        print("No videos given — pass one or more video paths to classify.")
        return 1

    for v in videos:
        try:
            windows, prob, pred, dt, fps = det.predict_video(v)
        except Exception as e:  # noqa
            print(f"  [FAIL] {v}: {e}")
            continue
        if windows is None:
            print(f"  {v}: no usable windows (pose rarely detected)")
            continue
        clip = os.path.splitext(os.path.basename(v))[0]
        print(f"  {os.path.basename(v)}")
        print(f"    windows={len(windows)} mean_prob={prob:.3f} "
              f"prediction={'FIDGET' if pred else 'NOT-FIDGET'} "
              f"(threshold {det.threshold:.2f}) "
              f"inference={dt:.2f}s over {len(windows)} windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
