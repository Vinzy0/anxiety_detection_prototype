import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import math
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


ANGLE_PATH = Path(r"F:\Thesis\videos\New folder\Thesis Data Gathering\Middle")
DATASET_ROOT = ANGLE_PATH.parent
OUTPUT_CSV = DATASET_ROOT / "symptom_frames.csv"
MODEL_DIR = Path(__file__).resolve().parent / "mediapipe_models"

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".flv", ".mpeg", ".mpg"}
ANGLES = {"left": "Left", "middle": "Middle", "right": "Right"}
SYMPTOMS = {
    "facial tension": "facial_tension",
    "hand tremor": "hand_tremor",
    "heavy breathing": "heavy_breathing",
    "knee tremor": "knee_tremor",
    "fidgeting": "fidgeting",
    "avoidant gaze": "avoidant_gaze",
    "fear": "fear",
}

FACE_IDS = sorted({
    1, 4, 10, 13, 14, 17, 33, 46, 52, 55, 61, 65, 70, 78, 105, 107,
    133, 145, 152, 159, 160, 168, 172, 234, 263, 276, 282, 285, 291,
    295, 300, 308, 334, 336, 362, 374, 386, 387, 398, 454,
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
})
BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward",
    "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "neutral", "noseSneerLeft", "noseSneerRight",
]

META_COLUMNS = [
    "angle", "participant_id", "symptom", "video_id", "frame", "timestamp_ms", "fps",
    "face_detected", "pose_detected", "body_reference_available",
    "left_hand_detected", "right_hand_detected",
]
LABEL_COLUMNS = [f"label_{slug}" for slug in SYMPTOMS.values()]
FACE_COLUMNS = [f"face_{i}_{axis}" for i in FACE_IDS for axis in ("x", "y", "z")]
BLEND_COLUMNS = [f"bs_{name}" for name in BLENDSHAPES]
POSE_COLUMNS = [f"pose_{i}_{feature}" for i in range(33) for feature in ("x", "y", "z", "visibility", "presence")]
HAND_COLUMNS = [f"{hand}_{i}_{axis}" for hand in ("left_hand", "right_hand") for i in range(21) for axis in ("x", "y", "z")]
CSV_COLUMNS = META_COLUMNS + LABEL_COLUMNS + FACE_COLUMNS + BLEND_COLUMNS + POSE_COLUMNS + HAND_COLUMNS


def clean_name(name):
    return " ".join(name.casefold().replace("_", " ").split())


def ensure_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for filename, url in MODELS.items():
        path = MODEL_DIR / filename
        if not path.exists() or path.stat().st_size == 0:
            urllib.request.urlretrieve(url, path)
        paths[filename] = path
    return paths


def make_landmarkers(model_paths):
    face = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_paths["face_landmarker.task"])),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    pose = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_paths["pose_landmarker.task"])),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    hand = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_paths["hand_landmarker.task"])),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return face, pose, hand


def find_videos():
    videos = []
    for angle_dir in DATASET_ROOT.iterdir():
        angle = ANGLES.get(clean_name(angle_dir.name)) if angle_dir.is_dir() else None
        if not angle:
            continue
        for symptom_dir in angle_dir.iterdir():
            symptom = SYMPTOMS.get(clean_name(symptom_dir.name)) if symptom_dir.is_dir() else None
            if not symptom:
                continue
            for participant_dir in symptom_dir.iterdir():
                if not participant_dir.is_dir():
                    continue
                for path in participant_dir.rglob("*"):
                    if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS:
                        videos.append({
                            "path": path,
                            "angle": angle,
                            "symptom": symptom,
                            "participant_id": participant_dir.name,
                            "video_id": path.relative_to(DATASET_ROOT).with_suffix("").as_posix().replace("/", "_"),
                        })
    return sorted(videos, key=lambda item: str(item["path"]).casefold())


def pixel_xyz(landmark, width, height):
    return landmark.x * width, landmark.y * height, landmark.z * width


def body_transform(pose, width, height):
    if not pose or len(pose) < 33:
        return None
    left_hip = pixel_xyz(pose[23], width, height)
    right_hip = pixel_xyz(pose[24], width, height)
    left_shoulder = pixel_xyz(pose[11], width, height)
    right_shoulder = pixel_xyz(pose[12], width, height)
    origin = tuple((left_hip[i] + right_hip[i]) / 2 for i in range(3))
    angle = math.atan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0])
    shoulder_width = math.hypot(right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1])
    shoulder_mid = tuple((left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3))
    torso_height = math.hypot(shoulder_mid[0] - origin[0], shoulder_mid[1] - origin[1])
    scale = max(shoulder_width, torso_height * 0.25, 1.0)
    return origin, angle, scale


def normalize_point(point, transform):
    origin, angle, scale = transform
    dx, dy, dz = (point[i] - origin[i] for i in range(3))
    c, s = math.cos(angle), math.sin(angle)
    return ((c * dx + s * dy) / scale, (-s * dx + c * dy) / scale, dz / scale)


def face_features(face, width, height):
    features = {column: 0.0 for column in FACE_COLUMNS}
    if not face or len(face) <= 362:
        return features

    left_inner = pixel_xyz(face[133], width, height)
    right_inner = pixel_xyz(face[362], width, height)
    left_outer = pixel_xyz(face[33], width, height)
    right_outer = pixel_xyz(face[263], width, height)
    center = tuple((left_inner[i] + right_inner[i]) / 2 for i in range(3))
    angle = math.atan2(right_inner[1] - left_inner[1], right_inner[0] - left_inner[0])
    scale = max(math.hypot(right_outer[0] - left_outer[0], right_outer[1] - left_outer[1]), 1.0)
    c, s = math.cos(angle), math.sin(angle)

    for index in FACE_IDS:
        if index >= len(face):
            continue
        x, y, z = pixel_xyz(face[index], width, height)
        dx, dy, dz = x - center[0], y - center[1], z - center[2]
        features[f"face_{index}_x"] = (c * dx + s * dy) / scale
        features[f"face_{index}_y"] = (-s * dx + c * dy) / scale
        features[f"face_{index}_z"] = dz / scale
    return features


def blendshape_features(blendshapes):
    features = {column: 0.0 for column in BLEND_COLUMNS}
    for item in blendshapes or []:
        name = f"bs_{getattr(item, 'category_name', '').lstrip('_')}"
        if name in features:
            features[name] = float(item.score or 0.0)
    return features


def pose_features(pose, width, height):
    features = {column: 0.0 for column in POSE_COLUMNS}
    transform = body_transform(pose, width, height)
    if transform is None:
        return features, None
    for i, landmark in enumerate(pose[:33]):
        x, y, z = normalize_point(pixel_xyz(landmark, width, height), transform)
        features.update({
            f"pose_{i}_x": x,
            f"pose_{i}_y": y,
            f"pose_{i}_z": z,
            f"pose_{i}_visibility": float(getattr(landmark, "visibility", 0.0) or 0.0),
            f"pose_{i}_presence": float(getattr(landmark, "presence", 0.0) or 0.0),
        })
    return features, transform


def hand_features(hand, prefix, width, height, transform):
    features = {f"{prefix}_{i}_{axis}": 0.0 for i in range(21) for axis in ("x", "y", "z")}
    if not hand or len(hand) < 21 or transform is None:
        return features
    for i, landmark in enumerate(hand[:21]):
        x, y, z = normalize_point(pixel_xyz(landmark, width, height), transform)
        features.update({f"{prefix}_{i}_x": x, f"{prefix}_{i}_y": y, f"{prefix}_{i}_z": z})
    return features


def assign_hands(result):
    hands = {"left": None, "right": None}
    unassigned = []
    for i, landmarks in enumerate(result.hand_landmarks or []):
        categories = result.handedness[i] if i < len(result.handedness) else []
        label = categories[0].category_name.casefold() if categories else ""
        if label in hands and hands[label] is None:
            hands[label] = landmarks
        else:
            unassigned.append(landmarks)
    for label in ("left", "right"):
        if hands[label] is None and unassigned:
            hands[label] = unassigned.pop(0)
    return hands["left"], hands["right"]


def symptom_labels(target):
    return {f"label_{slug}": int(slug == target) for slug in SYMPTOMS.values()}


def extract_video(video, writer, model_paths):
    cap = cv2.VideoCapture(str(video["path"]), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(video["path"]))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("OpenCV could not open this video or its codec is unsupported")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    face_task = pose_task = hand_task = None
    frames = 0
    try:
        face_task, pose_task, hand_task = make_landmarkers(model_paths)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            timestamp_ms = int(frames * 1000 / fps)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            face_result = face_task.detect_for_video(image, timestamp_ms)
            pose_result = pose_task.detect_for_video(image, timestamp_ms)
            hand_result = hand_task.detect_for_video(image, timestamp_ms)

            face = face_result.face_landmarks[0] if face_result.face_landmarks else None
            pose = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
            left_hand, right_hand = assign_hands(hand_result)
            pose_data, transform = pose_features(pose, width, height)

            row = {column: 0.0 for column in CSV_COLUMNS}
            row.update({
                "angle": video["angle"],
                "participant_id": video["participant_id"],
                "symptom": video["symptom"],
                "video_id": video["video_id"],
                "frame": frames,
                "timestamp_ms": timestamp_ms,
                "fps": fps,
                "face_detected": int(face is not None),
                "pose_detected": int(pose is not None),
                "body_reference_available": int(transform is not None),
                "left_hand_detected": int(left_hand is not None),
                "right_hand_detected": int(right_hand is not None),
            })
            row.update(symptom_labels(video["symptom"]))
            row.update(face_features(face, width, height))
            blendshapes = face_result.face_blendshapes[0] if face_result.face_blendshapes else []
            row.update(blendshape_features(blendshapes))
            row.update(pose_data)
            row.update(hand_features(left_hand, "left_hand", width, height, transform))
            row.update(hand_features(right_hand, "right_hand", width, height, transform))
            writer.writerow(row)
            frames += 1
    finally:
        cap.release()
        for task in (face_task, pose_task, hand_task):
            if task is not None:
                task.close()

    if frames == 0:
        raise RuntimeError("No frames were decoded; the video may be empty or use an unsupported codec")
    return frames


def main():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_ROOT}")

    required = (
        "FaceLandmarker", "FaceLandmarkerOptions",
        "PoseLandmarker", "PoseLandmarkerOptions",
        "HandLandmarker", "HandLandmarkerOptions",
    )
    missing = [name for name in required if not hasattr(mp_vision, name)]
    if missing:
        raise RuntimeError(
            f"Your MediaPipe installation lacks {', '.join(missing)}. "
            "Update it with: python -m pip install --upgrade mediapipe"
        )

    videos = find_videos()
    if not videos:
        raise FileNotFoundError(f"No .MOV, .MP4, or supported videos found under: {DATASET_ROOT}")

    model_paths = ensure_models()
    total_frames = 0
    failed = 0
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for video in videos:
            try:
                total_frames += extract_video(video, writer, model_paths)
            except Exception as error:
                failed += 1
                print(f"Failed: {video['path']} ({error})")

    print(f"Videos: {len(videos)} | Frames extracted: {total_frames} | Failed: {failed}")
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

