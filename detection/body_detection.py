import os
import numpy as np
from collections import deque
import mediapipe as mp

# Buffer sizes
RESTLESSNESS_BUFFER    = 150   # 5 seconds at ~30fps
BREATHING_BUFFER       = 300   # 10 seconds at ~30fps

# Restlessness: direction-reversal rate threshold (reversals per second)
RESTLESSNESS_THRESHOLD  = 1.5
RESTLESSNESS_MIN_DELTA  = 3.0    # pixels — ignore sub-pixel jitter from MediaPipe tracking

# Breathing: FFT-based frequency thresholds
BREATHING_FREQ_MIN  = 0.1    # Hz — floor of valid band (6 bpm)
BREATHING_FREQ_MAX  = 1.0    # Hz — ceiling of valid band (60 bpm); excludes deliberate fast pumps
BREATHING_THRESHOLD = 0.4    # Hz — >= this flags anxious breathing rate (24 bpm)
MIN_BREATHING_AMP   = 2.0    # FFT amplitude noise floor in pixel units
NOMINAL_FPS         = 30.0   # fallback if timestamp duration is zero

POSE_MODEL_PATH = 'pose_landmarker_lite.task'
POSE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
)

LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_WRIST     = 15
RIGHT_WRIST    = 16


def ensure_pose_model():
    if not os.path.exists(POSE_MODEL_PATH):
        print("Downloading pose landmarker model (~7MB)...")
        import urllib.request
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("Download complete.")


class BodyDetector:
    def __init__(self):
        ensure_pose_model()

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

        # Restlessness: track combined wrist activity relative to shoulder anchor
        self.arm_activity_history = deque(maxlen=RESTLESSNESS_BUFFER)
        self.arm_ts_history       = deque(maxlen=RESTLESSNESS_BUFFER)

        # Breathing: track shoulder height for FFT frequency analysis
        self.shoulder_history     = deque(maxlen=BREATHING_BUFFER)
        self.shoulder_ts_history  = deque(maxlen=BREATHING_BUFFER)

        self.restlessness_flagged = False
        self.breathing_flagged    = False
        self.restlessness_value   = 0.0
        self.breathing_value      = 0.0

    def update(self, rgb_frame, timestamp_ms):
        rgb_frame.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            h, w = rgb_frame.shape[:2]
            lm = result.pose_landmarks[0]

            ls = lm[LEFT_SHOULDER]
            rs = lm[RIGHT_SHOULDER]
            anchor_x = ((ls.x + rs.x) / 2) * w
            anchor_y = ((ls.y + rs.y) / 2) * h

            # Wrist positions relative to shoulder midpoint (cancels out body sways)
            lw = lm[LEFT_WRIST]
            rw = lm[RIGHT_WRIST]
            lw_rel = np.array([lw.x * w - anchor_x, lw.y * h - anchor_y])
            rw_rel = np.array([rw.x * w - anchor_x, rw.y * h - anchor_y])
            arm_activity = (np.linalg.norm(lw_rel) + np.linalg.norm(rw_rel)) / 2.0
            self.arm_activity_history.append(arm_activity)
            self.arm_ts_history.append(timestamp_ms)

            # Shoulder height for breathing frequency analysis
            shoulder_y = ((ls.y + rs.y) / 2) * h
            self.shoulder_history.append(shoulder_y)
            self.shoulder_ts_history.append(timestamp_ms)

        # Restlessness: direction-reversal rate over 5-second window.
        # Only count reversals among deltas above the noise floor — sub-pixel
        # jitter from MediaPipe tracking would otherwise produce constant sign
        # changes even when sitting completely still.
        if len(self.arm_activity_history) >= RESTLESSNESS_BUFFER:
            arr    = np.array(self.arm_activity_history)
            deltas = np.diff(arr)
            significant = deltas[np.abs(deltas) > RESTLESSNESS_MIN_DELTA]
            if len(significant) > 1:
                reversals = float(np.sum(np.diff(np.sign(significant)) != 0))
            else:
                reversals = 0.0
            elapsed_s = (self.arm_ts_history[-1] - self.arm_ts_history[0]) / 1000.0
            self.restlessness_value   = reversals / elapsed_s if elapsed_s > 0 else 0.0
            self.restlessness_flagged = self.restlessness_value > RESTLESSNESS_THRESHOLD

        # Breathing: FFT dominant frequency over 10-second window
        if len(self.shoulder_history) >= BREATHING_BUFFER:
            arr       = np.array(self.shoulder_history, dtype=np.float32)
            elapsed_s = (self.shoulder_ts_history[-1] - self.shoulder_ts_history[0]) / 1000.0
            fps       = (len(arr) - 1) / elapsed_s if elapsed_s > 0 else NOMINAL_FPS

            arr -= arr.mean()                      # remove DC offset
            arr_w = arr * np.hanning(len(arr))     # reduce spectral leakage

            fft_mag = np.abs(np.fft.rfft(arr_w))
            freqs   = np.fft.rfftfreq(len(arr), d=1.0 / fps)

            band_mask = (freqs >= BREATHING_FREQ_MIN) & (freqs <= BREATHING_FREQ_MAX)
            if np.any(band_mask):
                band_mags  = fft_mag[band_mask]
                band_freqs = freqs[band_mask]
                peak_idx   = np.argmax(band_mags)
                peak_freq  = float(band_freqs[peak_idx])
                peak_amp   = float(band_mags[peak_idx])
                self.breathing_value   = peak_freq
                self.breathing_flagged = (peak_freq >= BREATHING_THRESHOLD
                                          and peak_amp >= MIN_BREATHING_AMP)
            else:
                self.breathing_value   = 0.0
                self.breathing_flagged = False

        return (
            self.restlessness_flagged,
            self.breathing_flagged,
            self.restlessness_value,
            self.breathing_value,
            result
        )
