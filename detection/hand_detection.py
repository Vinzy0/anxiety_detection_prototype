import os
import numpy as np
from collections import deque
import mediapipe as mp

HISTORY_LENGTH = 10
JITTER_THRESHOLD = 8.0

HAND_MODEL_PATH = 'hand_landmarker.task'
HAND_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)


def ensure_hand_model():
    if not os.path.exists(HAND_MODEL_PATH):
        print("Downloading hand landmarker model (~7MB)...")
        import urllib.request
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("Download complete.")


class HandDetector:
    def __init__(self):
        ensure_hand_model()

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.position_history = deque(maxlen=HISTORY_LENGTH)
        self.flagged = False
        self.jitter_value = 0.0

    def update(self, rgb_frame, timestamp_ms):
        rgb_frame.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                wrist = hand_landmarks[0]
                h, w = rgb_frame.shape[:2]
                wrist_x = int(wrist.x * w)
                wrist_y = int(wrist.y * h)
                self.position_history.append((wrist_x, wrist_y))

        if len(self.position_history) >= HISTORY_LENGTH:
            positions = np.array(self.position_history)
            deltas = np.diff(positions, axis=0)
            distances = np.linalg.norm(deltas, axis=1)
            self.jitter_value = float(np.mean(distances))
            self.flagged = self.jitter_value > JITTER_THRESHOLD
        else:
            self.jitter_value = 0.0
            self.flagged = False

        return self.flagged, self.jitter_value, result
