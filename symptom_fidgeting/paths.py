"""Central path/config definitions for the fidgeting prototype.

All scripts import paths from here so the repo stays relocatable: only the
MAS dataset root is an external absolute path.
"""
import os

# Repo root = parent of this file's directory (symptom_fidgeting/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- External MAS dataset -------------------------------------------------
# Clips, annotations.json, and jsons/videos.json live under the official
# ScazLab clone checked out locally.
MAS_ROOT = r"C:\Users\vinzp\Documents\Codex\2026-08-29\i-need-you-to-handle-dataset\thesis-datasets\MAS\original\MAS_dataset"
MAS_CLIPS_DIR = os.path.join(MAS_ROOT, "MAS", "clips")
MAS_ANNOTATIONS = os.path.join(MAS_ROOT, "MAS", "annotations.json")
MAS_VIDEOS_JSON = os.path.join(MAS_ROOT, "jsons", "videos.json")

# --- MediaPipe asset -------------------------------------------------------
# Downloaded at runtime if missing (Lite, float16 — see README).
MEDIAPIPE_POSE_MODEL = os.path.join(
    REPO_ROOT, "mediapipe_models", "pose_landmarker_lite.task")
MEDIAPIPE_POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

# --- Outputs (inside repo; gitignored except models/) ----------------------
FIDGET_DIR = os.path.join(REPO_ROOT, "symptom_fidgeting")
EXTRACTED_DIR = os.path.join(FIDGET_DIR, "data_extracted")
EXPERIMENTS_DIR = os.path.join(FIDGET_DIR, "experiments")
MODELS_DIR = os.path.join(FIDGET_DIR, "models")

# Feature window used by extraction validation, training and inference.
WINDOW_FRAMES = 30
WINDOW_STEP = 5

# Decision threshold / window quality gates (single source of truth).
MIN_DETECTION_RATE = 0.60   # window needs >=60% pose-detected frames
MIN_DETECTED_FRAMES = 3     # window needs >=3 valid frames for motion features
