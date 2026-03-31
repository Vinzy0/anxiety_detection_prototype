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
        self.position_history = {
            "Left":  deque(maxlen=HISTORY_LENGTH),
            "Right": deque(maxlen=HISTORY_LENGTH),
        }
        self.flagged = False
        self.jitter_value = 0.0

    def update(self, rgb_frame, timestamp_ms):
        rgb_frame.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # NOTE: if two-hand tracking causes issues, fallback is to just use
        # result.hand_landmarks[0][0] and track only the first detected hand.
        # Tremors are physiological so both hands are affected anyway.
        if result.hand_landmarks:
            h, w = rgb_frame.shape[:2]
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].category_name  # "Left" or "Right"
                wrist = hand_landmarks[0]
                self.position_history[label].append((int(wrist.x * w), int(wrist.y * h)))

        hand_jitters = []
        for buf in self.position_history.values():
            if len(buf) >= HISTORY_LENGTH:
                positions = np.array(buf)
                deltas    = np.diff(positions, axis=0)
                hand_jitters.append(float(np.mean(np.linalg.norm(deltas, axis=1))))

        if hand_jitters:
            self.jitter_value = max(hand_jitters)
            self.flagged      = self.jitter_value > JITTER_THRESHOLD
        else:
            self.jitter_value = 0.0
            self.flagged      = False

        return self.flagged, self.jitter_value, result
