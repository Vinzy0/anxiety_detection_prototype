import cv2
import mediapipe as mp
import os
import time
import urllib.request

from detection.eye_detection import EyeDetector
from detection.mouth_detection import MouthDetector
from detection.hand_detection import HandDetector
from detection.body_detection import BodyDetector
from detection.symptom_checker import SymptomChecker, SYMPTOM_NAMES
from coping_tips import get_tip

MODEL_PATH = 'face_landmarker.task'
MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)

# Download the model file if it's not already here.
# Think of it like downloading a map before using GPS.
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model (~7MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
def draw_landmarks(image, face_landmarks_list):
    h, w = image.shape[:2]
    for face_landmarks in face_landmarks_list:
        for lm in face_landmarks:
            # MediaPipe gives coordinates as percentages (e.g. x=0.5 = halfway across).
            # Multiply by actual pixel width/height to get real pixel positions.
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,   # VIDEO = treat frames as a continuous stream, better tracking than treating each frame as a random photo
    num_faces=1,                            # only look for one face
    min_face_detection_confidence=0.5,      # only report a face if at least 50% sure it found one
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)  # 0 = first camera found (your webcam)
start_time = time.time()
eye_detector = EyeDetector()
mouth_detector = MouthDetector()
hand_detector = HandDetector()
body_detector = BodyDetector()
symptom_checker = SymptomChecker()

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)                                  # mirrors the image so it feels like a selfie camera
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # convert BGR -> RGB: webcam gives blue-green-red, MediaPipe expects red-green-blue

        timestamp_ms = int((time.time() - start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)  # send frame to MediaPipe, get back 478 face point locations

        flagged = False
        mouth_flagged = False

        if result.face_landmarks:
            draw_landmarks(frame, result.face_landmarks)

            # Get frame dimensions for eye detection (face_landmarks uses normalized 0-1 coords)
            h, w = frame.shape[:2]

            # Run eye detection — face_landmarks[0] is the first (and only) face
            flagged, ear, blink_count = eye_detector.update(result.face_landmarks[0], w, h)
            mouth_flagged, mar = mouth_detector.update(result.face_landmarks[0], w, h)

            # Display Eye Aspect Ratio (EAR) value
            # EAR ~0.25-0.30 when open, drops to ~0 when blinking
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display blink count within the 10-second window
            cv2.putText(frame, f"Blinks (10s): {blink_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # If rapid blinking is detected, show an alert in red
            if flagged:
                cv2.putText(frame, "RAPID BLINKING DETECTED", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"MAR: {mar:.3f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if mouth_flagged:
                cv2.putText(frame, "LIP COMPRESSION DETECTED", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        hand_flagged, jitter, hand_results = hand_detector.update(rgb_frame, timestamp_ms)

        if hand_results.hand_landmarks:
            h, w = frame.shape[:2]
            for hand_lms in hand_results.hand_landmarks:
                for lm in hand_lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        cv2.putText(frame, f"Hand jitter: {jitter:.1f}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if hand_flagged:
            cv2.putText(frame, "HAND TREMOR DETECTED", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        rest_flagged, breath_flagged, rest_val, breath_val, pose_results = body_detector.update(rgb_frame, timestamp_ms)

        if pose_results.pose_landmarks:
            h, w = frame.shape[:2]
            for lm in pose_results.pose_landmarks[0]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

        cv2.putText(frame, f"Restlessness: {rest_val:.1f}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Breathing Hz: {breath_val:.2f}", (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if rest_flagged:
            cv2.putText(frame, "RESTLESSNESS DETECTED", (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if breath_flagged:
            cv2.putText(frame, "RAPID BREATHING DETECTED", (10, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        anxiety_detected, active_symptoms = symptom_checker.update(
            flagged, mouth_flagged, hand_flagged, rest_flagged, breath_flagged
        )

        if anxiety_detected:
            symptom_labels = ", ".join(SYMPTOM_NAMES[s] for s in active_symptoms)
            cv2.putText(frame, f"ANXIETY DETECTED: {symptom_labels}", (10, 390),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            tip = get_tip(active_symptoms)
            cv2.putText(frame, tip, (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow('Anxiety Detection - Phase 5', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):  # check every 5ms if Q was pressed
            break

cap.release()
cv2.destroyAllWindows()
