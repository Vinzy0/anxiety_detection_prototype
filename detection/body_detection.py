import os
import numpy as np
from collections import deque
import mediapipe as mp

# Buffer sizes
RESTLESSNESS_BUFFER    = 150   # 5 seconds at ~30fps
BREATHING_BUFFER       = 300   # 10 seconds at ~30fps

# Restlessness: amplitude-gated direction-reversal detection
# MIN_RESTLESS_AMPLITUDE: minimum pixel displacement required to count a reversal.
#   Only movements exceeding this threshold are considered anxiety-related fidgeting.
#   Below this = normal drift/noise. Above this = intentional movement worth counting.
#   Tunable: lower = more sensitive, higher = less sensitive to small movements.
MIN_RESTLESS_AMPLITUDE  = 15     # pixels — amplitude gate for reversal counting

# RESTLESS_THRESHOLD: reversals per second that triggers the restlessness symptom.
#   Set to 3.0 now that amplitude gating filters out noise — previously 1.5 was too
#   sensitive because it counted all reversals including sub-threshold movements.
RESTLESS_THRESHOLD      = 3.0    # reversals per second — flags restlessness

# Head micro-movement tracking (secondary restlessness signal)
# Uses same amplitude-gated approach as arm restlessness.
HEAD_BUFFER             = 150    # 5 seconds at ~30fps
HEAD_MIN_AMPLITUDE      = 8      # pixels — head movements are smaller than arm movements
HEAD_THRESHOLD          = 3.0    # reversals per second — head fidgeting threshold (matches arm threshold)

# Breathing: FFT-based frequency thresholds
BREATHING_FREQ_MIN  = 0.1    # Hz — floor of valid band (6 bpm)
BREATHING_FREQ_MAX  = 1.0    # Hz — ceiling of valid band (60 bpm); excludes deliberate fast pumps
BREATHING_THRESHOLD = 0.4    # Hz — >= this flags anxious breathing rate (24 bpm)
MIN_BREATHING_AMP   = 2.0    # FFT amplitude noise floor in pixel units
NOMINAL_FPS         = 30.0   # fallback if timestamp duration is zero

# Stage 1 — per-frame moving-average window applied BEFORE the FFT buffer.
# Averages the last BREATH_SMOOTH_WINDOW raw Y-positions so that frame-to-frame
# landmark jitter is suppressed before the smoothed value enters the FFT signal.
BREATH_SMOOTH_WINDOW  = 7    # frames — width of the upstream smoothing window

# Stage 2 — ZCR tolerance around the expected 2× multiplier.
# For a true breathing cycle at f Hz the signal crosses its mean ~2f times/sec.
# We accept the FFT reading when ZCR falls within [f×(2−tol), f×(2+tol)].
BREATH_ZCR_TOLERANCE  = 0.5  # tunable — widens/narrows the acceptance band

POSE_MODEL_PATH = 'pose_landmarker_lite.task'
POSE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
)

LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
NOSE_TIP       = 0   # nose tip landmark (index 0 in pose landmarks)


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

        # Head micro-movement: track nose tip Y position for restlessness
        self.head_history         = deque(maxlen=HEAD_BUFFER)
        self.head_ts_history      = deque(maxlen=HEAD_BUFFER)

        # Breathing: track shoulder height for FFT frequency analysis
        self.shoulder_history     = deque(maxlen=BREATHING_BUFFER)
        self.shoulder_ts_history  = deque(maxlen=BREATHING_BUFFER)

        # Stage 1 smoothing buffer: holds the last BREATH_SMOOTH_WINDOW raw Y values.
        # The mean of this buffer is what gets pushed into shoulder_history each frame,
        # so only already-smoothed values ever reach the FFT.
        self.breath_smooth_buffer = deque(maxlen=BREATH_SMOOTH_WINDOW)

        self.restlessness_flagged      = False
        self.breathing_flagged         = False
        self.restlessness_value        = 0.0
        self.breathing_value           = 0.0
        self.last_valid_breathing_value = 0.0  # held when ZCR validation rejects FFT

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

            # Shoulder height for breathing frequency analysis.
            # Stage 1: push raw Y into the smoothing window, then append the window
            # mean to the FFT buffer. This kills frame-to-frame landmark jitter while
            # preserving the slow breathing wave (~0.2–0.5 Hz).
            shoulder_y = ((ls.y + rs.y) / 2) * h
            self.breath_smooth_buffer.append(shoulder_y)
            smoothed_y = float(np.mean(self.breath_smooth_buffer))
            self.shoulder_history.append(smoothed_y)
            self.shoulder_ts_history.append(timestamp_ms)

            # Head micro-movement: track nose tip Y position relative to shoulder anchor
            # Nose is landmark 0 in MediaPipe Pose
            nose_tip = lm[NOSE_TIP]
            nose_y = nose_tip.y * h - anchor_y
            self.head_history.append(nose_y)
            self.head_ts_history.append(timestamp_ms)

        # Restlessness: amplitude-gated direction-reversal detection
        # 
        # ALGORITHM: Peak-tracking with amplitude gating
        # 1. Iterate through the arm_activity history (scalar values representing
        #    average wrist distance from shoulder anchor)
        # 2. Track direction changes: when the signal switches from increasing to
        #    decreasing or vice versa, that's a potential peak
        # 3. At each direction change, calculate the amplitude (distance from
        #    current value to previous peak value)
        # 4. ONLY count it as a valid reversal if amplitude >= MIN_RESTLESS_AMPLITUDE
        # 5. This filters out tiny positional drift while catching real fidgeting
        #
        # WHY: Without amplitude gating, even sub-pixel noise from MediaPipe tracking
        # causes constant direction changes, triggering false positives. The amplitude
        # gate ensures only movements large enough to be visually noticeable count.
        if len(self.arm_activity_history) >= RESTLESSNESS_BUFFER:
            arr = np.array(self.arm_activity_history)
            elapsed_s = (self.arm_ts_history[-1] - self.arm_ts_history[0]) / 1000.0
            
            # Peak-tracking state
            last_value = arr[0]           # Value at previous frame
            last_direction = 0            # 1 = increasing, -1 = decreasing, 0 = unknown
            peak_value = arr[0]           # Value at last counted reversal
            valid_reversals = 0           # Count of amplitude-gated reversals
            
            for i in range(1, len(arr)):
                current_value = arr[i]
                delta = current_value - last_value
                
                # Determine current direction (ignore tiny movements < 1px to avoid noise)
                if abs(delta) < 1.0:
                    current_direction = last_direction  # No significant change, keep previous direction
                elif delta > 0:
                    current_direction = 1   # Increasing
                else:
                    current_direction = -1  # Decreasing
                
                # Detect direction change (potential peak)
                if last_direction != 0 and current_direction != last_direction:
                    # Direction changed! Calculate amplitude from last peak
                    amplitude = abs(current_value - peak_value)
                    
                    # AMPLITUDE GATE: Only count reversal if movement was significant
                    # This is the key fix — filters out small positional drift
                    if amplitude >= MIN_RESTLESS_AMPLITUDE:
                        valid_reversals += 1
                        peak_value = current_value  # Update peak only on valid reversal
                
                last_value = current_value
                last_direction = current_direction
            
            # Calculate reversals per second
            arm_reversals_per_sec = valid_reversals / elapsed_s if elapsed_s > 0 else 0.0
        else:
            arm_reversals_per_sec = 0.0

        # Head micro-movement: amplitude-gated direction-reversal detection
        # Uses same peak-tracking algorithm as arm restlessness, but with smaller
        # amplitude threshold (HEAD_MIN_AMPLITUDE = 8px) since head movements are subtler.
        # Tracks nose tip Y position relative to shoulder anchor.
        if len(self.head_history) >= HEAD_BUFFER:
            arr = np.array(self.head_history)
            elapsed_s = (self.head_ts_history[-1] - self.head_ts_history[0]) / 1000.0
            
            # Peak-tracking state (same algorithm as arms)
            last_value = arr[0]
            last_direction = 0
            peak_value = arr[0]
            valid_reversals = 0
            
            for i in range(1, len(arr)):
                current_value = arr[i]
                delta = current_value - last_value
                
                # Determine current direction (ignore tiny movements < 0.5px for head)
                if abs(delta) < 0.5:
                    current_direction = last_direction
                elif delta > 0:
                    current_direction = 1
                else:
                    current_direction = -1
                
                # Detect direction change (potential peak)
                if last_direction != 0 and current_direction != last_direction:
                    amplitude = abs(current_value - peak_value)
                    
                    # AMPLITUDE GATE: Use HEAD_MIN_AMPLITUDE (8px) for head movements
                    if amplitude >= HEAD_MIN_AMPLITUDE:
                        valid_reversals += 1
                        peak_value = current_value
                
                last_value = current_value
                last_direction = current_direction
            
            head_reversals_per_sec = valid_reversals / elapsed_s if elapsed_s > 0 else 0.0
        else:
            head_reversals_per_sec = 0.0

        # Combine arm and head restlessness (either can trigger)
        # Uses RESTLESS_THRESHOLD (3.0) for arms and HEAD_THRESHOLD (3.0) for head
        self.restlessness_value   = max(arm_reversals_per_sec, head_reversals_per_sec)
        self.restlessness_flagged = (arm_reversals_per_sec > RESTLESS_THRESHOLD or
                                     head_reversals_per_sec > HEAD_THRESHOLD)

        # Breathing: FFT dominant frequency over 10-second window.
        # Guard: fewer than 30 frames means the buffer just started filling —
        # return 0.0 and suppress the flag to avoid startup false positives.
        if len(self.shoulder_history) < 30:
            self.breathing_value   = 0.0
            self.breathing_flagged = False
        elif len(self.shoulder_history) >= BREATHING_BUFFER:
            # shoulder_history already contains Stage-1-smoothed values (mean of
            # BREATH_SMOOTH_WINDOW raw frames), so no batch convolution is needed here.
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

                if peak_amp < MIN_BREATHING_AMP:
                    # Amplitude too low for FFT to be meaningful — use ZCR estimate
                    # directly (counts half-cycles, divides by 2 to get full breaths/sec).
                    zero_crossings  = np.sum(np.diff(np.sign(arr)) != 0) / 2.0
                    breaths_per_sec = zero_crossings / elapsed_s if elapsed_s > 0 else 0.0
                    self.breathing_value   = breaths_per_sec
                    self.breathing_flagged = breaths_per_sec > BREATHING_THRESHOLD
                else:
                    # Stage 2 — ZCR validation of the FFT reading.
                    # A true breathing cycle at f Hz crosses the signal mean ~2f times/sec
                    # (once rising, once falling). We compute the actual ZCR and check
                    # whether it falls in the acceptance band [f×(2−tol), f×(2+tol)].
                    # If it does, the FFT and time-domain signals agree → accept.
                    # If it doesn't, noise is likely dominating → hold the last accepted value.
                    zero_crossings = np.sum(np.diff(np.sign(arr)) != 0)
                    zcr = zero_crossings / elapsed_s if elapsed_s > 0 else 0.0

                    zcr_lo = peak_freq * (2.0 - BREATH_ZCR_TOLERANCE)  # = peak_freq × 1.5
                    zcr_hi = peak_freq * (2.0 + BREATH_ZCR_TOLERANCE)  # = peak_freq × 2.5

                    if zcr_lo <= zcr <= zcr_hi:
                        # ZCR agrees with FFT — this is a reliable reading
                        self.breathing_value            = peak_freq
                        self.last_valid_breathing_value = peak_freq
                        self.breathing_flagged          = peak_freq >= BREATHING_THRESHOLD
                    else:
                        # ZCR disagrees — FFT is likely picking up jitter or artefact.
                        # Hold the last accepted value so the display stays stable.
                        self.breathing_value   = self.last_valid_breathing_value
                        self.breathing_flagged = (self.last_valid_breathing_value
                                                  >= BREATHING_THRESHOLD)
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
