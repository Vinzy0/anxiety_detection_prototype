import os
import numpy as np
from collections import deque
import mediapipe as mp
import cv2
from scipy.signal import find_peaks
import time

POSE_MODEL_PATH = 'pose_landmarker_lite.task'
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

def ensure_pose_model(): # Checks if the pose landmarker model file exists, and downloads it if not.
    if not os.path.exists(POSE_MODEL_PATH):
        print("Downloading pose landmarker model (~7MB)...")
        import urllib.request
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("Download complete.")

FPS                  = 30     # Frames per second your webcam captures
HISTORY_SECONDS      = 10     # How many seconds of shoulder data to keep in memory
BREATHING_ALERT_BPM  = 20     # Breaths/min above this = elevated
VISIBILITY_MIN       = 0.5    # Ignore landmarks below this confidence (0.0-1.0)
DISPLAY_WIDTH        = 1280   # Webcam window width in pixels
DISPLAY_HEIGHT       = 720    # Webcam window height in pixels
BREATHING_THRESHOLD = 0.4 # Minimum breathing value (in hz)

HISTORY_LEN = FPS * HISTORY_SECONDS

# Indexes for left and right shoulder landmarks in the MediaPipe pose model
LEFT_SHOULDER_IDX  = 11
RIGHT_SHOULDER_IDX = 12

class BodyDetector:
    def __init__(self):
        ensure_pose_model()

        BaseOptions = mp.tasks.BaseOptions # Tells mediapipe WHERE the model file is on disk.
        PoseLandmarker = mp.tasks.vision.PoseLandmarker # The actual ML model class we will use
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions # Config settings for the model, like input size and thresholds
        VisionRunningMode = mp.tasks.vision.RunningMode # For live camera input

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1, # Only detect one person at a time
            min_tracking_confidence=0.5 # 60% sure that the landmarks are accurate enough to keep tracking
        )


        # Create the actual detector object using our settings above.
        self.landmarker = PoseLandmarker.create_from_options(options)

        # Shoulder y-positions for the last 10 seconds
        # Inhale = 0.0, Exhale = 1.0
        self.shoulder_y_history = deque(maxlen=HISTORY_LEN)

        self.breathing_flagged    = False
        self.breathing_value      = 0.0

        self.last_result = None

    def compute_bpm(self):
        if len(self.shoulder_y_history) < HISTORY_LEN:
            return None  # Not enough data yet to compute a reliable BPM

        signal = np.array(self.shoulder_y_history)
        signal = signal - np.mean(signal) # Average the signal to avoid false peaks from overall height differences on the frame
        signal = -signal # Invert the signal because not moving is 0.0. Inhaling = shoulders go up, which is a negative change in y-value

        min_distance = FPS * 1.5 # Minimum gap between breaths = 1.5 secs
        peaks, _ = find_peaks(signal, distance=min_distance, prominence=0.01) # Finds peaks (inhales) in the shoulder movement signal. Prominence = filters out small peaks
        
        if len(peaks) < 2:
            return None  # Not enough peaks to calculate BPM
        
        intervals = np.diff(peaks) / FPS
        bpm = 60 / float(np.mean(intervals)) # Converts the average breath intervals to bpm (breaths per minute)

        self.breathing_value = round(bpm, 1)
        self.breathing_flagged = self.breathing_value >= (BREATHING_THRESHOLD * 60) # Breathing gets flagged when over 20 bpm (* 60 cuz it's in hz and we're converting it to bpm)

        return self.breathing_value

    def process_frame(self, frame, rgb_frame, timestamp_ms):
        rgb_frame.flags.writeable = False # Improves performance by telling mediapipe we won't modify the input image data
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) # Convert the RGB frame from OpenCV into mediapipe format
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms) # Run the pose landmarker model on the current video frame
        rgb_frame.flags.writeable = True # Allow modifications to the RGB frame again

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0] # Get the first detected person's landmarks
            ls = landmarks[LEFT_SHOULDER_IDX] # Left shoulder landmark
            rs = landmarks[RIGHT_SHOULDER_IDX] # Right shoulder landmark

            # Draw circles on the detected shoulder landmarks for visualization
            for lm in (ls, rs):
                if lm.visibility >= VISIBILITY_MIN: # Only draw if the landmark is confidently detected
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]) # Convert normalized coordinates to pixel values
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
            visible_y = [lm.y for lm in (ls, rs) if lm.visibility >= VISIBILITY_MIN]   # Only use this reading if one of the shoulders is visible
            if len(visible_y):
                avg_y = sum(visible_y) / len(visible_y)
                self.shoulder_y_history.append(avg_y) # Store the average shoulder height in our history buffer for the last 10 seconds
        
        return self.compute_bpm(), frame
    
    @property
    def is_warming_up(self):
        return len(self.shoulder_y_history) < FPS * 10 # First 10 seconds after starting the app is warmup time for the body detector, during which we don't trust the readings as much
    
    @property
    def warmup_progress(self):
        return len(self.shoulder_y_history) / (FPS * 10) # Returns a value from 0.0 to 1.0 indicating how far through the warmup period we are
