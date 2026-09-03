"""MediaPipe Pose Landmarker helpers shared by extractor and inference."""
import os

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

from . import paths

# Pose landmark indices used by the fidgeting prototype (MediaPipe Pose
# topology, 33 landmarks). Upper-body joints relevant to fidgeting.
# Landmark coordinates are normalized [0..1] to the frame.
LM = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_pinky": 17, "r_pinky": 18,
    "l_index": 19, "r_index": 20,
    "l_thumb": 21, "r_thumb": 22,
    "l_hip": 23, "r_hip": 24,
}

# All named joints we extract (subset relevant to upper-body fidgeting).
EXTRACT_JOINTS = ["l_wrist", "r_wrist", "l_elbow", "r_elbow",
                  "l_shoulder", "r_shoulder", "l_hip", "r_hip",
                  "l_index", "r_index", "l_thumb", "r_thumb",
                  "nose"]

# Relative-motion reference pairs (joint, base) used in feature engineering.
# Moves of the joint relative to its base are the classic fidget signals.
REFERENCE_PAIRS = [
    ("l_wrist", "l_shoulder"), ("r_wrist", "r_shoulder"),
    ("l_elbow", "l_shoulder"), ("r_elbow", "r_shoulder"),
    ("l_shoulder", "nose"), ("r_shoulder", "nose"),
    ("l_hip", "nose"), ("r_hip", "nose"),
]

# Scale reference: shoulder width in normalized units (used to make motion
# features invariant to camera distance / body size).
SCALE_LANDMARKS = ("l_shoulder", "r_shoulder")


def _col(joint, axis):
    """Column name for a joint's normalized coordinate axis, e.g.
    lm_l_wrist_x."""
    return f"lm_{joint}_{axis}"


def landmark_dict(pose_landmarks):
    """Standard per-frame landmark dict from a MediaPipe Pose result.

    Keys: lm_<joint>_{x,y} (rounded), lm_conf_<joint> (visibility), and
    scale_ref (shoulder width). Used identically by the extractor and the
    inference path so both produce the exact same DataFrame columns.
    """
    d = {}
    for j in EXTRACT_JOINTS:
        lm = pose_landmarks[LM[j]]
        d[_col(j, "x")] = round(lm.x, 6)
        d[_col(j, "y")] = round(lm.y, 6)
        d[CONF_PREFIX + j] = round(lm.visibility, 5)
    d["scale_ref"] = round(
        abs(pose_landmarks[LM["l_shoulder"]].x
            - pose_landmarks[LM["r_shoulder"]].x), 6)
    return d

# 0/1 column value meaning "pose detected on this frame" (kept as float to
# match the fear pipeline's CSV convention).
DETECTED_COL = "pose_detected"
CONF_PREFIX = "lm_conf_"
CONF_COLS = ("lm_conf_l_wrist", "lm_conf_r_wrist")


def ensure_model():
    """Download the official MediaPipe Pose Landmarker (Lite) if missing."""
    if not os.path.exists(paths.MEDIAPIPE_POSE_MODEL):
        os.makedirs(os.path.dirname(paths.MEDIAPIPE_POSE_MODEL), exist_ok=True)
        import urllib.request
        print(f"Downloading Pose Landmarker Lite -> "
              f"{paths.MEDIAPIPE_POSE_MODEL}")
        urllib.request.urlretrieve(paths.MEDIAPIPE_POSE_URL,
                                   paths.MEDIAPIPE_POSE_MODEL)
    return paths.MEDIAPIPE_POSE_MODEL


def make_pose_landmarker(min_detection_confidence=0.5,
                         min_presence_confidence=0.5,
                         min_tracking_confidence=0.5):
    """Create a VIDEO-mode PoseLandmarker (single person, upper body)."""
    model_path = ensure_model()
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)
