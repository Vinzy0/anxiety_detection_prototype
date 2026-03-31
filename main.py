import cv2
import mediapipe as mp
import os
import time
import threading
import urllib.request

from detection.eye_detection import EyeDetector
from detection.mouth_detection import MouthDetector
from detection.hand_detection import HandDetector
from detection.body_detection import BodyDetector
from detection.symptom_checker import SymptomChecker

from coping_tips import get_tip
from ui.display import draw_symptom_panel
from ui.settings_panel import launch_settings_panel

MODEL_PATH = 'face_landmarker.task'
MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model (~7MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode


def draw_landmarks(image, face_landmarks_list):
    h, w = image.shape[:2]
    for face_landmarks in face_landmarks_list:
        for lm in face_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(image, (x, y), 1, (0, 200, 0), -1)


def camera_loop():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap        = cv2.VideoCapture(0)
    start_time = time.time()

    eye_detector    = EyeDetector()
    mouth_detector  = MouthDetector()
    hand_detector   = HandDetector()
    body_detector   = BodyDetector()
    symptom_checker = SymptomChecker()

    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Could not read from webcam.")
                break

            frame        = cv2.flip(frame, 1)
            rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w         = frame.shape[:2]
            timestamp_ms = int((time.time() - start_time) * 1000)

            # Default metric values in case face / hands are not detected this frame
            flagged       = False
            mouth_flagged = False
            ear           = 0.0
            blink_count   = 0
            mar           = 0.0

            # ── Face ──────────────────────────────────────────────────────────
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result   = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                draw_landmarks(frame, result.face_landmarks)
                flagged,       ear, blink_count = eye_detector.update(result.face_landmarks[0], w, h)
                mouth_flagged, mar              = mouth_detector.update(result.face_landmarks[0], w, h)

            # ── Hands ──────────────────────────────────────────────────────────
            hand_flagged, jitter, hand_results = hand_detector.update(rgb_frame, timestamp_ms)

            if hand_results.hand_landmarks:
                for hand_lms in hand_results.hand_landmarks:
                    for lm in hand_lms:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 200, 0), -1)

            # ── Body ───────────────────────────────────────────────────────────
            rest_flagged, breath_flagged, rest_val, breath_val, pose_results = body_detector.update(rgb_frame, timestamp_ms)

            if pose_results.pose_landmarks:
                for lm in pose_results.pose_landmarks[0]:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 200, 0), -1)

            # ── Symptom checker ────────────────────────────────────────────────
            anxiety_detected, active_symptoms = symptom_checker.update(
                flagged, mouth_flagged, hand_flagged, rest_flagged, breath_flagged
            )

            tip = get_tip(active_symptoms) if anxiety_detected else ""

            # ── UI panel ───────────────────────────────────────────────────────
            metrics = [
                ("Blinks / 10s",   float(blink_count), 10.0),
                ("Restlessness",   rest_val,             3.0),
                ("Breathing (Hz)", breath_val,           0.8),
                ("Hand jitter",    jitter,               16.0),
            ]

            frame = draw_symptom_panel(frame, active_symptoms, anxiety_detected, tip, metrics)

            cv2.imshow('Anxiety Detection', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Run the camera loop in a background thread so tkinter can own the main thread
t = threading.Thread(target=camera_loop, daemon=True)
t.start()

# Settings panel runs on the main thread (required by tkinter on Windows)
launch_settings_panel()
