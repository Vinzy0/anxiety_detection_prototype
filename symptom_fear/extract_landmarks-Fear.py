import os
import csv
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


# PATHS

CREMA_VIDEO_DIR = r"F:\Github\CREMA-D\VideoFlash"
OUTPUT_DIR      = r"F:\Thesis\videos\data\landmarks_cremad_fear"
MEDIAPIPE_MODEL = "mediapipe_models/face_landmarker.task"

ALLOWED_EMOTIONS = ["FEA","NEU"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Removed wanted Blendshapes, cause we might need all of them.

# Removed subject discovery, this dataset doesn't have folders like SN001/ and such.

# Removed AU4 loading as well, this dataset doesn't have AU


def parse_cremad_filename(filename: str):
    parts = os.path.splitext(os.path.basename(filename))[0].split("_")

    if len(parts) < 4:
        return None

    actor, sentence, emotion, intensity = parts

    return {
        "actor": actor,
        "sentence": sentence,
        "emotion": emotion,
        "intensity": intensity
    }

# MEDIAPIPE SETUP

def make_face_landmarker():
    if not os.path.exists(MEDIAPIPE_MODEL):
        raise FileNotFoundError(
            f"MediaPipe model not found: {MEDIAPIPE_MODEL}\n"
            f"Run 01_download_models.py first."
        )

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,   # not needed, saves compute
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


# VIDEO PROCESSING
# lm_out is now video_id
# label_out is now output_path
def process_video(video_path: str, label: int,
                  metadata: dict, output_path: str) -> int:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Cannot open: {video_path}")
        return 0

    # DISFA is 30 FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30   # safe fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    {os.path.basename(video_path)}: "
          f"{total_frames} frames @ {fps:.2f} FPS, ")

    # Fresh landmarker per video
    landmarker = make_face_landmarker()

    rows   = []
    labels = []
    frame_idx = 0
    # The selected blendshapes below correspond to facial movements commonly
    # associated with facial tension and discomfort. Limiting extraction to
    # relevant blendshapes reduces storage requirements while preserving the
    # most informative expression features.

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        ts_ms = int(frame_idx * 1000.0 / fps)

        # MediaPipe requires RGB. OpenCV loads BGR.
        # Skipping this conversion causes colour-channel confusion in the
        # neural network, significantly reducing detection accuracy.
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_img, ts_ms)

        row = {"frame": frame_idx}

        row = {
            "dataset": "CREMA-D",
            "actor": metadata["actor"],
            "video_id": os.path.splitext(os.path.basename(video_path))[0],
            "sentence": metadata["sentence"],
            "emotion": metadata["emotion"],
            "intensity": metadata["intensity"],
            "label": label,
            "frame": frame_idx,
            "timestamp_ms": ts_ms
        }

        if result.face_landmarks:
            lms = result.face_landmarks[0]

            # Changed to all-landmarks so this can be used as a template

            for idx, lm in enumerate(lms):
                row[f"lm_{idx}_x"] = round(lm.x, 6)
                row[f"lm_{idx}_y"] = round(lm.y, 6)
                row[f"lm_{idx}_z"] = round(lm.z, 6)

            row["face_detected"] = 1.0

            # Extract all blendshapes.
            if result.face_blendshapes:
                for cat in result.face_blendshapes[0]:
                    row[f"bs_{cat.category_name}"] = round(cat.score, 5)
        else:
            # Face not detected this frame.
            # Store 0.0 for everything.
            row["face_detected"] = 0.0

        rows.append(row)

        frame_idx += 1

    cap.release()
    landmarker.close()

    if not rows:
        print(f"    No frames extracted.")
        return 0

    # All extracted landmarks and blendshape values are stored as CSV files.
    # CSV storage simplifies downstream processing because the training stage
    # can directly load structured numerical features without reprocessing the
    # original videos.

    all_keys = sorted(set(k for r in rows for k in r.keys()))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, restval=0.0)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return frame_idx



# BASELINE COMPUTATION

# Changed subject_id to actor
def compute_baseline(actor: str):

    base_path = os.path.join(OUTPUT_DIR, f"{actor}_baseline.csv")
    if os.path.exists(base_path):
        print(f"    Baseline ({actor}): already exists, skipping.")
        return
 
    actor_files = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith(f"{actor}_") and f.endswith(".csv")
        and not f.endswith("_baseline.csv")
    ]
 
    neutral_frames = []
    for fname in actor_files:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
        if "emotion" not in df.columns:
            continue
        neutral_frames.append(df[df["emotion"] == "NEU"])
 
    if not neutral_frames:
        print(f"    Baseline skipped ({actor}): no NEU clips found.")
        return
 
    neutral_df = pd.concat(neutral_frames, ignore_index=True).fillna(0.0)
 
    if len(neutral_df) < 30:
        print(f"    WARNING: {actor} has only {len(neutral_df)} "
              f"neutral frames. Writing zero baseline.")
        zero_baseline = pd.DataFrame([{col: 0.0 for col in neutral_df.columns}])
        zero_baseline.to_csv(base_path, index=False)
        return
 
    numeric_df = neutral_df.select_dtypes(include="number")
    baseline = numeric_df.mean().to_frame().T
    baseline.to_csv(base_path, index=False)
 
    print(f"    Baseline ({actor}): computed from {len(neutral_df)} "
          f"neutral frames. Saved to {os.path.basename(base_path)}")
 

def process_all_videos():
    video_files = sorted(
        f for f in os.listdir(CREMA_VIDEO_DIR)
        if f.lower().endswith(".flv")
    )

    processed = 0

    for filename in video_files:
        metadata = parse_cremad_filename(filename)

        if metadata is None:
            print(f"Skipping invalid filename: {filename}")
            continue

        if metadata["emotion"] not in ALLOWED_EMOTIONS:
            continue

        label = 1 if metadata["emotion"] == "FEA" else 0

        video_path = os.path.join(CREMA_VIDEO_DIR, filename)
        video_id = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{video_id}.csv")

        if os.path.exists(output_path):
            continue

        process_video(
            video_path,
            label,
            metadata,
            output_path
        )


        processed += 1


    return processed

# PER-SUBJECT PROCESSING
# Commented out for now, as CREMA doesn't have two angles. In the future it is best we do three angles when recording.

# def process_subject(subject_id: str) -> int:

#     # Load AU4 labels once, same annotations apply to both camera views
#     au4_values = load_au4_labels(subject_id)
#     if not au4_values:
#         print(f"  {subject_id}: no AU4 labels, skipping entirely.")
#         return 0

#     total_frames = 0

#     for side_name, side_key in [("Left", "left"), ("Right", "right")]:
#         video_path = os.path.join(
#             DISFA_VIDEO_DIR, f"{side_name}Video{subject_id}_comp.avi"
#         )
#         lm_out    = os.path.join(OUTPUT_DIR, f"{subject_id}_{side_key}_landmarks.csv")
#         label_out = os.path.join(OUTPUT_DIR, f"{subject_id}_{side_key}_labels.csv")

#         if not os.path.exists(video_path):
#             print(f"    {side_name} video not found: "
#                   f"{os.path.basename(video_path)}, skipping.")
#             continue

#         # Skip landmark extraction if both output files already exist.
#         if os.path.exists(lm_out) and os.path.exists(label_out):
#             print(f"    {side_name}: landmark + label CSVs exist, skipping extraction.")
#         else:
#             n = process_video(video_path, au4_values, lm_out, label_out)
#             total_frames += n

#         compute_baseline(subject_id, side_key)

#     return total_frames


# MAIN
# The main section validates the required files and directories before starting
# extraction. Each discovered subject is processed sequentially and converted
# into landmark and label CSV files for later model training.

if __name__ == "__main__":
    print("=" * 60)
    print("CREMA-D FEAR EXTRACTION")

    # Pre-flight checks 
    errors = []
    if not os.path.exists(MEDIAPIPE_MODEL):
        errors.append(
            f"MediaPipe model not found: {MEDIAPIPE_MODEL}\n"
            f"  Fix: python 01_download_models.py"
        )

    if not os.path.exists(CREMA_VIDEO_DIR):
        errors.append(
            f"CREMA-D VideoData not found: {CREMA_VIDEO_DIR}\n"
            f"  Fix: check CREMA_VIDEO_DIR at the top of this file"
        )

    if errors:
        for e in errors:
            print(f"\nERROR: {e}")
        exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    count = process_all_videos()

    print(f"Videes processed: {count}")

    print(f"\n{'='*60}")
    print(f"Extraction complete.")
    print(f"\nFiles now in {OUTPUT_DIR}:")


