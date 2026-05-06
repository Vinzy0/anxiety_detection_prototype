import cv2
import mediapipe as mp
import os
import time
import threading
import urllib.request

from detection.facial_detection import FacialTensionDetector
from detection.hand_detection import HandDetector, HISTORY_LENGTH
from detection.body_detection import FPS, BodyDetector
from detection.symptom_checker import SymptomChecker

from coping_tips import COPING_TIPS
from ui.display import draw_symptom_panel, _text, MUTED
from ui.settings_panel import launch_settings_panel
from logger import AnxietyLogger

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def camera_loop():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    facial_detector = FacialTensionDetector(
    os.path.join(BASE_DIR, "tflite/facial_tension.tflite"),
    os.path.join(BASE_DIR, "tflite/scaler_mean.npy"),
    os.path.join(BASE_DIR, "tflite/scaler_std.npy")
    )

    cap           = cv2.VideoCapture(0)
    start_time    = time.time()
    tip_index     = 0
    last_tip_time = time.time()
    TIP_INTERVAL  = 15  # seconds per tip

    hand_detector   = HandDetector()
    body_detector   = BodyDetector()
    symptom_checker = SymptomChecker()
    anxiety_logger  = AnxietyLogger()

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

            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            face_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row = {}

            if face_result.face_landmarks:
                lms = face_result.face_landmarks[0]

                for idx in [55, 285, 159, 386]:
                    row[f"lm{idx}_x"] = lms[idx].x
                    row[f"lm{idx}_y"] = lms[idx].y

                for lm in lms:
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)

                    cv2.circle(
                        frame,
                        (x_px, y_px),
                        2,
                        (0, 200, 0),
                        -1
                    )

                row["face_detected"] = 1.0

                if face_result.face_blendshapes:
                    for cat in face_result.face_blendshapes[0]:
                        row[f"bs_{cat.category_name}"] = cat.score
            else:
                row["face_detected"] = 0.0
            
            face_flagged = False
            face_prob = 0.0

            facial_result = facial_detector.predict(row)

            if facial_result is not None:
                face_flagged = facial_result["tense"]
                face_prob = facial_result["probability"]

            facial_result = facial_detector.predict(row)

            if facial_result is None:
                face_flagged = False
                face_prob = 0.0
            else:
                face_flagged = facial_result["tense"]
                face_prob = facial_result["probability"]

            # ── Hands ──────────────────────────────────────────────────────────
            hand_flagged, jitter, hand_results = hand_detector.update(rgb_frame, timestamp_ms)

            if hand_results.hand_landmarks:
                for hand_lms in hand_results.hand_landmarks:
                    for lm in hand_lms:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 200, 0), -1)

            # ── Body ───────────────────────────────────────────────────────────
            breath_val, frame = body_detector.process_frame(frame, rgb_frame, timestamp_ms)

            if breath_val is None:
                breath_val    = 0.0
                breath_flagged = False
            else:
                breath_flagged = body_detector.breathing_flagged

            # ── Symptom checker ────────────────────────────────────────────────
            anxiety_detected, active_symptoms = symptom_checker.update(
                hand_flagged, face_flagged, False, breath_flagged
            )

            anxiety_logger.update(anxiety_detected, active_symptoms)

            now = time.time()
            if now - last_tip_time >= TIP_INTERVAL:
                tip_index     = (tip_index + 1) % len(COPING_TIPS)
                last_tip_time = now
            tip = COPING_TIPS[tip_index]

            # ── UI panel ───────────────────────────────────────────────────────
            if hand_detector.buffer_progress < HISTORY_LENGTH:
                hand_label = f"Hand tremor (warmup {hand_detector.buffer_progress}/{HISTORY_LENGTH})"
                hand_val   = float(hand_detector.buffer_progress)
                hand_max   = float(HISTORY_LENGTH)
            else:
                hand_label = "Hand tremor"
                hand_val   = jitter
                hand_max   = 100.0

            warmup = body_detector.is_warming_up
            warmup_remaining = max(0, 10 - len(body_detector.shoulder_y_history) // FPS)
            metrics = [
                ("Facial tension", face_prob, 1.0),
                ("Breathing", breath_val, 20.0),
                (hand_label,       hand_val,   hand_max),
            ]

            frame = draw_symptom_panel(frame, active_symptoms, anxiety_detected, tip, metrics, warmup=warmup, warmup_remaining=warmup_remaining)

            cv2.imshow('Symptom Monitor', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Run the camera loop in a background thread so tkinter can own the main thread
t = threading.Thread(target=camera_loop, daemon=True)
t.start()

# Settings panel runs on the main thread (required by tkinter on Windows)
launch_settings_panel()
