# Press Q to quit, SPACE to pause.
# Usage (Type the stuff below in terminal):
# python visualize_model.py --video path/to/video.mp4
# python visualize_model.py --video path/to/video.mp4 --subject SN003

import argparse
import os
import glob
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import tensorflow as tf

#  Paths, edit if needed 
MEDIAPIPE_MODEL  = "mediapipe_models/face_landmarker.task"
TFLITE_MODEL     = "output/facial_tension.tflite"
SCALER_MEAN_PATH = "output/scaler_mean.npy"
SCALER_STD_PATH  = "output/scaler_std.npy"
LANDMARKS_DIR    = r"F:\Thesis\videos\data\landmarks"


WINDOW_FRAMES = 30
N_FEATURES    = 28
STD_INDICES   = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 22, 26}

WANTED_BS = {
    "browDownLeft", "browDownRight", "browInnerUp",
    "eyeSquintLeft", "eyeSquintRight", "cheekSquintLeft",
    "jawOpen", "mouthPressLeft", "mouthPressRight",
}


# Feature extraction

def stat(arr):
    a = np.asarray(arr, np.float32)
    return float(np.mean(a)), float(np.std(a))


def extract_window_features(rows):
    """rows = list of dicts from MediaPipe results."""
    if not rows:
        return None

    def col(key, default=0.0):
        return [r.get(key, default) for r in rows]

    det_rate = float(np.mean(col("face_detected", 1.0)))
    if det_rate < 0.60:
        return None

    bg_l   = [r.get("lm159_y", 0) - r.get("lm55_y", 0) for r in rows]
    bg_r   = [r.get("lm386_y", 0) - r.get("lm285_y", 0) for r in rows]
    bg_avg = [(a + b) / 2 for a, b in zip(bg_l, bg_r)]

    ibd  = [abs(r.get("lm285_x", 0) - r.get("lm55_x", 0)) for r in rows]

    bdl  = col("bs_browDownLeft")
    bdr  = col("bs_browDownRight")
    biu  = col("bs_browInnerUp")
    esl  = col("bs_eyeSquintLeft")
    esr  = col("bs_eyeSquintRight")
    jaw  = col("bs_jawOpen")
    mpl  = col("bs_mouthPressLeft")
    mpr  = col("bs_mouthPressRight")

    bt   = [(a + b) / 2 - 0.5 * c for a, b, c in zip(bdl, bdr, biu)]
    comp = [(a + b + c + d + (1 - e) + f) / 6
            for a, b, c, d, e, f in zip(esl, esr, mpl, mpr, jaw, bt)]

    bg_l_m,   bg_l_s   = stat(bg_l)
    bg_r_m,   bg_r_s   = stat(bg_r)
    bg_avg_m, bg_avg_s = stat(bg_avg)
    ibd_m,    ibd_s    = stat(ibd)
    bdl_m,    bdl_s    = stat(bdl)
    bdr_m,    bdr_s    = stat(bdr)
    biu_m,    biu_s    = stat(biu)
    bt_m,     bt_s     = stat(bt)
    esl_m,    esl_s    = stat(esl)
    esr_m,    esr_s    = stat(esr)
    es_avg_m, _        = stat([(a + b) / 2 for a, b in zip(esl, esr)])
    jaw_m,    jaw_s    = stat(jaw)
    mpl_m,    _        = stat(mpl)
    mpr_m,    _        = stat(mpr)
    comp_m,   comp_s   = stat(comp)

    return [
        bg_l_m,   bg_l_s,
        bg_r_m,   bg_r_s,
        bg_avg_m, bg_avg_s,
        ibd_m,    ibd_s,
        bdl_m,    bdl_s,
        bdr_m,    bdr_s,
        biu_m,    biu_s,
        bt_m,     bt_s,
        esl_m,    esl_s,
        esr_m,    esr_s,
        es_avg_m,
        jaw_m,    jaw_s,
        mpl_m,    mpr_m,
        comp_m,   comp_s,
        det_rate,
    ]


# Load subject z-score baseline

def load_subject_baseline(subject_id):
    """
    Load per-subject z-score params from existing landmark CSVs.
    Returns (mean_vec, std_vec) or (zeros, ones) if not found.
    """
    # Try left camera first, then right
    # This will be different for the actual one, this is for DISFA
    for side in ["left", "right"]:
        lm_path = os.path.join(LANDMARKS_DIR, f"{subject_id}_{side}_landmarks.csv")
        lb_path = os.path.join(LANDMARKS_DIR, f"{subject_id}_{side}_labels.csv")
        if not os.path.exists(lm_path) or not os.path.exists(lb_path):
            continue

        df     = pd.read_csv(lm_path).fillna(0.0)
        labels = pd.read_csv(lb_path)["label"].values
        n      = min(len(df), len(labels))
        df, labels = df.iloc[:n], labels[:n]

        det_col      = df.get("face_detected", pd.Series([1.0]*n))
        neutral_mask = (labels == 0) & (det_col.values > 0.5)
        neutral_df   = df[neutral_mask]

        if len(neutral_df) < 30:
            continue

        # Build neutral feature windows
        windows = []
        for start in range(0, len(neutral_df) - WINDOW_FRAMES, 10):
            w_rows = []
            for idx in range(start, start + WINDOW_FRAMES):
                row = neutral_df.iloc[idx].to_dict()
                w_rows.append(row)
            f = extract_window_features(w_rows)
            if f is not None:
                windows.append(f)

        if len(windows) < 5:
            continue

        arr   = np.array(windows, np.float32)
        mean_v = arr.mean(axis=0)
        std_v  = np.clip(arr.std(axis=0), 0.001, None)
        print(f"  Loaded z-score baseline for {subject_id} ({side}): "
              f"{len(windows)} neutral windows")
        return mean_v, std_v

    print(f"  No baseline found for {subject_id}. Using raw features.")
    return np.zeros(N_FEATURES, np.float32), np.ones(N_FEATURES, np.float32)


# Drawing helpers

def draw_bar(img, x, y, w, h, value, label, color_high, color_low=(50, 200, 50)):
    """Draw a horizontal bar indicator."""
    value = float(np.clip(value, 0, 1))
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    bar_w = int(w * value)
    color = color_high if value > 0.5 else color_low
    cv2.rectangle(img, (x, y), (x + bar_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
    cv2.putText(img, f"{label}: {value:.3f}", (x + w + 8, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)


def draw_overlay(frame, rows, feats, prob, subject_baseline):
    """Draw all visualization on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent sidebar
    panel_w = 340
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    px = w - panel_w + 10

    # Probability meter
    prob_color = (0, 0, 220) if prob > 0.58 else (0, 200, 100)
    label_text = "TENSE" if prob > 0.58 else "RELAXED"
    cv2.putText(frame, f"FACIAL TENSION MODEL", (px, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.putText(frame, label_text, (px, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255) if prob > 0.58 else (0, 220, 80), 2)

    bar_h = 22
    draw_bar(frame, px, 68, panel_w - 80, bar_h, prob,
             "prob", (0, 0, 255), (0, 180, 80))

    if rows:
        last = rows[-1]
        cv2.putText(frame, "── Raw blendshapes (last frame) ──", (px, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        bs_items = [
            ("browDownLeft",  last.get("bs_browDownLeft",  0)),
            ("browDownRight", last.get("bs_browDownRight", 0)),
            ("browInnerUp",   last.get("bs_browInnerUp",   0)),
            ("eyeSquintL",    last.get("bs_eyeSquintLeft", 0)),
            ("eyeSquintR",    last.get("bs_eyeSquintRight",0)),
            ("jawOpen",       last.get("bs_jawOpen",        0)),
            ("mouthPressL",   last.get("bs_mouthPressLeft", 0)),
        ]
        for idx, (name, val) in enumerate(bs_items):
            y_pos = 124 + idx * 28
            draw_bar(frame, px, y_pos, 130, 18, val,
                     name[:14], (30, 100, 220), (30, 140, 30))

        # Geometric features
        cv2.putText(frame, "── Geometry (window mean) ──", (px, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        if feats:
            geo_items = [
                ("brow gap L",   feats[0] * 5),
                ("brow gap R",   feats[2] * 5),
                ("inner brow d", feats[6] * 3),
                ("brow tension", np.clip((feats[14] + 0.3) / 0.6, 0, 1)),
                ("eye squint",   feats[20]),
                ("composite",    np.clip((feats[26] + 0.2) / 0.5, 0, 1)),
            ]
            for idx, (name, val) in enumerate(geo_items):
                y_pos = 342 + idx * 28
                draw_bar(frame, px, y_pos, 130, 18, float(np.clip(val, 0, 1)),
                         name, (180, 60, 60), (30, 140, 30))

        # Z-score info
        baseline_applied = not np.all(subject_baseline[0] == 0)
        cv2.putText(frame,
                    f"Z-score: {'applied' if baseline_applied else 'none'}",
                    (px, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    # Face landmark dots
    if rows and rows[-1].get("face_detected", 0):
        last = rows[-1]
        lm_pts = {
            55:  (int(last.get("lm55_x",  0) * (w - panel_w)),
                  int(last.get("lm55_y",  0) * h)),
            285: (int(last.get("lm285_x", 0) * (w - panel_w)),
                  int(last.get("lm285_y", 0) * h)),
            159: (int(last.get("lm159_x", 0) * (w - panel_w))
                  if "lm159_x" in last else 0,
                  int(last.get("lm159_y", 0) * h)),
            386: (int(last.get("lm386_x", 0) * (w - panel_w))
                  if "lm386_x" in last else 0,
                  int(last.get("lm386_y", 0) * h)),
        }
        colors = {55: (0, 255, 255), 285: (0, 255, 255),
                  159: (255, 100, 100), 386: (255, 100, 100)}
        labels_lm = {55: "L-brow", 285: "R-brow",
                     159: "L-eye", 386: "R-eye"}

        for idx, (px_pt, py_pt) in lm_pts.items():
            if px_pt > 0 and py_pt > 0:
                cv2.circle(frame, (px_pt, py_pt), 6, colors[idx], -1)
                cv2.putText(frame, labels_lm[idx],
                            (px_pt + 8, py_pt - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            colors[idx], 1)

    # Window fill indicator
    fill = len(rows) / WINDOW_FRAMES
    cv2.putText(frame, f"Buffer: {len(rows)}/{WINDOW_FRAMES}",
                (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.rectangle(frame, (10, h - 10), (10 + int(200 * fill), h - 4),
                  (80, 180, 80), -1)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   required=True,
                        help="Path to video file (.mp4 or .avi)")
    parser.add_argument("--subject", default=None,
                        help="DISFA subject ID for z-score (e.g. SN003)")
    parser.add_argument("--loop",    action="store_true",
                        help="Loop the video")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        return

    # Load model
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    interp = tf.lite.Interpreter(model_path=TFLITE_MODEL)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    scaler_mean = np.load(SCALER_MEAN_PATH)
    scaler_std  = np.load(SCALER_STD_PATH)

    # Load subject baseline 
    subject_baseline = (np.zeros(N_FEATURES, np.float32),
                        np.ones(N_FEATURES,  np.float32))
    if args.subject:
        subject_baseline = load_subject_baseline(args.subject)

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {os.path.basename(args.video)}")
    print(f"  FPS: {fps:.1f}  |  Frames: {total}")
    print(f"  Subject baseline: "
          f"{'applied (' + args.subject + ')' if args.subject else 'none'}")
    print("\nControls: Q=quit  SPACE=pause  R=restart")

    frame_idx  = 0
    rows_buf   = []
    prob       = 0.0
    feats_disp = None
    paused     = False

    while True:
        if not paused:
            ret, bgr = cap.read()
            if not ret:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    rows_buf.clear()
                    landmarker.close()
                    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
                    continue
                else:
                    print("Video ended. Press Q to quit or R to restart.")
                    paused = True
                    continue

            ts_ms = int(frame_idx * 1000.0 / fps)
            rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            # Build row dict
            row = {}
            if result.face_landmarks:
                lms = result.face_landmarks[0]
                for idx in [55, 285, 159, 386]:
                    row[f"lm{idx}_x"] = lms[idx].x
                    row[f"lm{idx}_y"] = lms[idx].y
                row["face_detected"] = 1.0
                if result.face_blendshapes:
                    for cat in result.face_blendshapes[0]:
                        if cat.category_name in WANTED_BS:
                            row[f"bs_{cat.category_name}"] = cat.score
            else:
                row["face_detected"] = 0.0

            rows_buf.append(row)
            if len(rows_buf) > WINDOW_FRAMES:
                rows_buf.pop(0)

            # Run inference when buffer full
            if len(rows_buf) == WINDOW_FRAMES:
                feats = extract_window_features(rows_buf)
                if feats is not None:
                    feats_disp = feats[:]

                    # Apply per-subject z-score
                    feat_arr = np.array(feats, np.float32)
                    mean_v, std_v = subject_baseline
                    for fi in range(N_FEATURES):
                        if fi not in STD_INDICES:
                            feat_arr[fi] = (feat_arr[fi] - mean_v[fi]) / std_v[fi]

                    # Apply global scaler
                    feat_scaled = (feat_arr - scaler_mean) / scaler_std
                    feat_scaled = feat_scaled.astype(np.float32)

                    # TFLite inference
                    interp.set_tensor(inp_d['index'], feat_scaled[np.newaxis, :])
                    interp.invoke()
                    prob = float(interp.get_tensor(out_d['index'])[0][0])

            frame_idx += 1

        # Resize for display if very large
        disp = bgr.copy() if not paused else disp
        dh, dw = disp.shape[:2]
        max_w = 1200
        if dw > max_w:
            scale = max_w / dw
            disp  = cv2.resize(disp, (max_w, int(dh * scale)))

        disp = draw_overlay(disp, rows_buf, feats_disp, prob, subject_baseline)

        # Frame counter
        cv2.putText(disp, f"Frame {frame_idx}/{total}  {'[PAUSED]' if paused else ''}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Facial Tension — MediaPipe + TFLite", disp)

        # Playback speed, match video FPS
        wait_ms = max(1, int(1000 / fps))
        key = cv2.waitKey(1 if paused else wait_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            rows_buf.clear()
            prob = 0.0
            feats_disp = None
            paused = False
            landmarker.close()
            landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()