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

DISFA_VIDEO_DIR = r"F:\Thesis\videos\data\raw\disfa\VideoData"
DISFA_AU_DIR    = r"F:\Thesis\videos\data\raw\disfa\ActionUnit_Labels"
OUTPUT_DIR      = r"F:\Thesis\videos\data\landmarks"
MEDIAPIPE_MODEL = "mediapipe_models/face_landmarker.task"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Blendshape names extracted from MediaPipe.
# These are the only ones needed for the 28 features.
# All other blendshapes are discarded to keep CSV files small.
WANTED_BS = {
    "browDownLeft", "browDownRight", "browInnerUp",
    "eyeSquintLeft", "eyeSquintRight", "cheekSquintLeft",
    "jawOpen", "mouthPressLeft", "mouthPressRight",
}


# SUBJECT DISCOVERY
# The subject folders are detected automatically from the DISFA annotation
# directory. This prevents the need to manually hardcode subject identifiers
# and allows the extraction stage to adapt if subjects are added or removed.
# Automatic discovery also reduces the possibility of processing errors caused
# by incorrect subject lists.

def get_available_subjects() -> list[str]:

    # Read subject IDs from the ActionUnit_Labels folder.
    # Returns sorted list: ['SN001', 'SN002', 'SN003', ...]

    if not os.path.exists(DISFA_AU_DIR):
        raise FileNotFoundError(
            f"ActionUnit_Labels folder not found: {DISFA_AU_DIR}\n"
            f"Check that DISFA_AU_DIR is set correctly."
        )
    subjects = []
    for name in os.listdir(DISFA_AU_DIR):
        full = os.path.join(DISFA_AU_DIR, name)
        # Valid subject folder: is a directory, starts with SN, 5 chars (SN001)
        if os.path.isdir(full) and name.startswith("SN") and len(name) == 5:
            try:
                int(name[2:])   # confirm SN### format (digits after SN)
                subjects.append(name)
            except ValueError:
                continue
    return sorted(subjects)

# AU4 LABEL LOADING
# AU4 corresponds to brow lowering, which is strongly associated with facial
# tension, discomfort, and frustration. The labels are loaded frame by frame
# from the DISFA annotation files. These labels serve as the ground truth used
# during supervised training.
#
# The extraction stage converts the AU intensity values into binary labels.
# Frames containing detectable brow lowering are treated as tense samples,
# while frames without brow lowering are treated as relaxed samples.

def load_au4_labels(subject_id: str) -> list[int]:

    path = os.path.join(DISFA_AU_DIR, subject_id, f"{subject_id}_au4.txt")

    if not os.path.exists(path):
        print(f"    AU4 file not found: {path}")
        return []

    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                values.append(int(parts[-1].strip()))
            except ValueError:
                continue   # skip malformed lines

    return values


# MEDIAPIPE SETUP
# MediaPipe Face Landmarker is initialized in VIDEO mode so that landmark
# tracking remains temporally consistent between consecutive frames. VIDEO mode
# improves landmark stability because the detector uses previous frame
# information to guide subsequent detections.
#
# Blendshape extraction is enabled as blendshapes represent semantic facial
# movements such as brow lowering, eye squinting, and jaw movement. These
# expression related features are more informative for facial tension analysis
# than geometric coordinates alone.

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
# Each DISFA subject video is processed frame by frame. The extraction stage
# converts every detected face into numerical landmark coordinates and
# blendshape scores. The resulting values are stored in CSV format so the
# training stage can operate on compact structured features instead of raw
# images.
#
# The study uses only the frontal camera view because it most closely matches
# the intended deployment condition of a mobile front facing camera. Using only
# the frontal perspective improves consistency between the training data and
# real world application conditions.

def process_video(video_path: str, au4_values: list[int],
                  lm_out: str, label_out: str) -> int:

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
          f"{total_frames} frames @ {fps:.2f} FPS, "
          f"{len(au4_values)} AU4 labels")

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

        if result.face_landmarks:
            lms = result.face_landmarks[0]

            # Four landmarks needed for brow geometry.
            # All MediaPipe coordinates are normalised [0, 1] by frame size.
            # This makes them resolution-independent and scale-invariant.
            #   55  = left inner brow corner
            #   285 = right inner brow corner
            #   159 = left eye center (iris)
            #   386 = right eye center (iris)
            for idx in [55, 285, 159, 386]:
                row[f"lm{idx}_x"] = round(lms[idx].x, 6)
                row[f"lm{idx}_y"] = round(lms[idx].y, 6)

            row["face_detected"] = 1.0

            # Extract only the blendshapes used.
            if result.face_blendshapes:
                for cat in result.face_blendshapes[0]:
                    if cat.category_name in WANTED_BS:
                        row[f"bs_{cat.category_name}"] = round(cat.score, 5)
        else:
            # Face not detected this frame.
            # Store 0.0 for everything.
            row["face_detected"] = 0.0

        rows.append(row)

        # Binary label from AU4 intensity.
        # AU4 >= 1 = any detectable brow lowering = tense (1)
        # AU4 == 0 = no brow lowering = relaxed (0)

        # Frames with AU4 intensity greater than or equal to 1 are treated as
        # tense samples. A lower threshold increases the number of positive
        # training examples and improves the model's ability to learn subtle
        # tension related facial movements.

        if frame_idx < len(au4_values):
            labels.append(1 if au4_values[frame_idx] >= 1 else 0)
        else:
            labels.append(0)

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
    with open(lm_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, restval=0.0)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write label CSV, one row per frame, single column label
    with open(label_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"])
        for v in labels[:frame_idx]:
            writer.writerow([v])

    tense   = sum(labels[:frame_idx])
    det     = sum(1 for r in rows if r.get("face_detected", 0) == 1.0)
    print(f"    -> tense={tense} ({tense/frame_idx*100:.1f}%)  "
          f"face detected={det} ({det/frame_idx*100:.1f}%)")

    return frame_idx



# BASELINE COMPUTATION

def compute_baseline(subject_id: str, side: str):
    # This procedure computes a neutral baseline for each subject using
    # frames labeled as AU4 = 0. The purpose of this step is to reduce
    # inter-subject variation caused by natural facial structure and
    # resting facial expression differences. Since some individuals may
    # naturally exhibit higher or lower eyebrow positions even in a
    # relaxed state, directly using absolute blendshape values may cause
    # the model to associate subject-specific facial structure with
    # tension-related behavior. To address this issue, the mean values
    # of the neutral frames are computed and stored as the subject's
    # baseline representation. This allows the succeeding stages of the
    # system to interpret facial movement as deviation from the subject's
    # own resting state rather than relying solely on absolute values.

    #TLDR: Adjusts what is neutral for each person.

    lm_path    = os.path.join(OUTPUT_DIR, f"{subject_id}_{side}_landmarks.csv")
    label_path = os.path.join(OUTPUT_DIR, f"{subject_id}_{side}_labels.csv")
    base_path  = os.path.join(OUTPUT_DIR, f"{subject_id}_{side}_baseline.csv")

    if not os.path.exists(lm_path) or not os.path.exists(label_path):
        print(f"    Baseline skipped ({side}): landmark/label CSV missing.")
        return

    if os.path.exists(base_path):
        print(f"    Baseline ({side}): already exists, skipping.")
        return

    df     = pd.read_csv(lm_path).fillna(0.0)
    labels = pd.read_csv(label_path)["label"].values
    n      = min(len(df), len(labels))
    df, labels = df.iloc[:n], labels[:n]

    # Select only AU4=0 frames, the person's neutral/relaxed state.
    neutral_mask = (labels == 0)
    neutral_df   = df[neutral_mask]

    if len(neutral_df) < 30:
        # Too few neutral frames to compute a reliable baseline.
        # Write a zero baseline so training code can still read the file.
        print(f"    WARNING: {subject_id} {side} has only {len(neutral_df)} "
              f"neutral frames. Writing zero baseline.")
        zero_baseline = pd.DataFrame([{col: 0.0 for col in df.columns}])
        zero_baseline.to_csv(base_path, index=False)
        return

    # Column-wise mean of neutral frames = this subject's resting face values.
    baseline = neutral_df.mean().to_frame().T
    baseline.to_csv(base_path, index=False)

    print(f"    Baseline ({side}): computed from {len(neutral_df)} neutral frames. "
          f"Saved to {os.path.basename(base_path)}")



# PER-SUBJECT PROCESSING

def process_subject(subject_id: str) -> int:
    # This procedure processes both the left and right camera recordings
    # provided in the DISFA dataset. The use of both camera perspectives
    # increases the amount of training data available to the model while
    # also exposing the system to slight variations in facial orientation.
    # Although the left camera more closely resembles the frontal angle
    # commonly encountered in mobile front-facing cameras, incorporating
    # the right camera improves the robustness of the model against small
    # pose variations that may occur during real-world usage.

    # Load AU4 labels once, same annotations apply to both camera views
    au4_values = load_au4_labels(subject_id)
    if not au4_values:
        print(f"  {subject_id}: no AU4 labels, skipping entirely.")
        return 0

    total_frames = 0

    for side_name, side_key in [("Left", "left"), ("Right", "right")]:
        video_path = os.path.join(
            DISFA_VIDEO_DIR, f"{side_name}Video{subject_id}_comp.avi"
        )
        lm_out    = os.path.join(OUTPUT_DIR, f"{subject_id}_{side_key}_landmarks.csv")
        label_out = os.path.join(OUTPUT_DIR, f"{subject_id}_{side_key}_labels.csv")

        if not os.path.exists(video_path):
            print(f"    {side_name} video not found: "
                  f"{os.path.basename(video_path)}, skipping.")
            continue

        # Skip landmark extraction if both output files already exist.
        if os.path.exists(lm_out) and os.path.exists(label_out):
            print(f"    {side_name}: landmark + label CSVs exist, skipping extraction.")
        else:
            n = process_video(video_path, au4_values, lm_out, label_out)
            total_frames += n

        compute_baseline(subject_id, side_key)

    return total_frames


# MAIN
# The main section validates the required files and directories before starting
# extraction. Each discovered subject is processed sequentially and converted
# into landmark and label CSV files for later model training.

if __name__ == "__main__":
    print("=" * 60)
    print("DISFA LANDMARK EXTRACTION Both Cameras + Baselines")
    print("=" * 60)

    # Pre-flight checks 
    errors = []
    if not os.path.exists(MEDIAPIPE_MODEL):
        errors.append(
            f"MediaPipe model not found: {MEDIAPIPE_MODEL}\n"
            f"  Fix: python 01_download_models.py"
        )
    if not os.path.exists(DISFA_VIDEO_DIR):
        errors.append(
            f"DISFA VideoData not found: {DISFA_VIDEO_DIR}\n"
            f"  Fix: check DISFA_VIDEO_DIR at the top of this file"
        )
    if not os.path.exists(DISFA_AU_DIR):
        errors.append(
            f"DISFA ActionUnit_Labels not found: {DISFA_AU_DIR}\n"
            f"  Fix: check DISFA_AU_DIR at the top of this file"
        )
    if errors:
        for e in errors:
            print(f"\nERROR: {e}")
        exit(1)

    # Discover subjects 
    subjects = get_available_subjects()
    if not subjects:
        print("No subject folders found. Check DISFA_AU_DIR.")
        exit(1)

    print(f"\nFound {len(subjects)} subjects: {subjects[0]} … {subjects[-1]}")
    print(f"Processing: LEFT + RIGHT camera per subject")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"\nNote: existing landmark/label CSVs are skipped automatically.")
    print(f"      Only missing baseline CSVs will be newly created.\n")

    # Process all subjects
    total_new_frames = 0
    total_subjects   = len(subjects)

    for i, subj in enumerate(subjects, 1):
        print(f"\n[{i}/{total_subjects}] {subj}:")
        n = process_subject(subj)
        total_new_frames += n

    # Summary
    # Count what now exists in the output directory
    lm_files    = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith("_landmarks.csv")])
    label_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith("_labels.csv")])
    base_files  = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith("_baseline.csv")])

    print(f"\n{'='*60}")
    print(f"Extraction complete.")
    print(f"  New frames extracted this run: {total_new_frames:,}")
    print(f"\nFiles now in {OUTPUT_DIR}:")
    print(f"  Landmark CSVs : {lm_files}   (expect {len(subjects) * 2})")
    print(f"  Label CSVs    : {label_files}  (expect {len(subjects) * 2})")
    print(f"  Baseline CSVs : {base_files}   (expect {len(subjects) * 2})")

    if base_files < len(subjects) * 2:
        missing = len(subjects) * 2 - base_files
        print(f"\n  WARNING: {missing} baseline CSVs are missing.")

