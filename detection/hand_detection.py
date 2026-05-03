import os
import numpy as np
from collections import deque
import mediapipe as mp

# Module-level constants below are written by the settings panel (main thread)
# and read by the camera loop (daemon thread). Writes are locked via
# settings_panel._settings_lock. Reads are not locked — safe under CPython's GIL
# for simple scalar assignment, but not guaranteed under GIL-free runtimes.

HISTORY_LENGTH = 64
JITTER_THRESHOLD = 8.0  # unused/deprecated — kept for backward compatibility
# JITTER_THRESHOLD removed — superseded by FFT-based tremor detection (MIN_TREMOR_AMP)

TREMOR_FREQ_MIN = 7.5   # 0.5 Hz buffer below clinical 8 Hz lower bound to absorb FFT bin-edge quantization
TREMOR_FREQ_MAX = 12.0
MIN_TREMOR_AMP = 50.0
# TODO: threshold was calibrated against single-bin amplitude ratio (old formula).
# Now uses band_power/total_power (magnitude²). Needs empirical re-tuning — 0.35
# may be too high or too low under the new scale. Run qa_tremor_fft.py after tuning.
TREMOR_RELATIVE_POWER_THRESHOLD = 0.35

# Sustained window: fraction of recent analysis windows that must show tremor
# to trigger the flag. Prevents false positives from transient bursts.
# At 30fps, maxlen=300 holds ~10 seconds of per-frame history.
TREMOR_SUSTAINED_RATIO = 0.5

# If a hand disappears for more than this many frames, clear its buffer so that
# old timestamps don't corrupt the FPS calculation when the hand returns.
HAND_LOSS_RESET_FRAMES = 10

DEBUG = False  # set to True for frame-by-frame console diagnostics

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


def _analyze_tremor_buffer(positions, timestamps):
    """
    Run the FFT tremor detection pipeline on a single hand's buffer.

    Operates on raw POSITION (where the hand is), not displacement (how much
    it moved each frame). This prevents big slow sweeping movements from
    triggering false positives: a slow sweep is a low-frequency position signal
    with no energy at 8-12 Hz, whereas displacement of a constant-speed sweep
    is a square wave whose harmonics bleed into the tremor band.

    Linear detrending removes any steady drift or sweep before the FFT, leaving
    only the oscillatory content (tremor) in the signal.

    Parameters
    ----------
    positions : list or array of shape (N, 2)
        Raw (x, y) fingertip or wrist pixel coordinates.
    timestamps : list or array of shape (N,)
        Frame timestamps in milliseconds.

    Returns
    -------
    tuple (tremor_detected, peak_amp, peak_freq)
        tremor_detected : bool
        peak_amp        : float — FFT magnitude of the dominant peak
        peak_freq       : float — frequency (Hz) of the dominant peak
    """
    positions = np.array(positions, dtype=np.float64)  # shape (N, 2)
    N = len(positions)

    x = positions[:, 0]
    y = positions[:, 1]

    # Linear detrend each axis — removes steady sweeps and slow drifts.
    # Subtracts the straight line from the first to the last sample,
    # leaving only oscillations (tremor) around that trend.
    t_idx = np.arange(N, dtype=np.float64)
    x -= np.polyval(np.polyfit(t_idx, x, 1), t_idx)
    y -= np.polyval(np.polyfit(t_idx, y, 1), t_idx)

    # Remove any remaining DC offset
    x -= np.mean(x)
    y -= np.mean(y)

    # Hanning window to reduce spectral leakage
    window = np.hanning(N)
    x_w = x * window
    y_w = y * window

    # Compute real FPS from elapsed timestamps — webcam FPS varies under load
    elapsed_s = (timestamps[-1] - timestamps[0]) / 1000.0
    real_fps = (N - 1) / elapsed_s if elapsed_s > 0 else 30.0

    freqs = np.fft.rfftfreq(N, d=1.0 / real_fps)
    band_mask = (freqs >= TREMOR_FREQ_MIN) & (freqs <= TREMOR_FREQ_MAX)

    if not np.any(band_mask):
        return False, 0.0, 0.0

    band_freqs = freqs[band_mask]

    # FFT on each axis separately — picks up tremor even if only one axis
    # is visible (e.g. hand seen edge-on due to camera angle)
    fft_x = np.abs(np.fft.rfft(x_w))
    fft_y = np.abs(np.fft.rfft(y_w))

    band_x = fft_x[band_mask]
    band_y = fft_y[band_mask]

    peak_amp_x = float(np.max(band_x))
    peak_amp_y = float(np.max(band_y))

    # Use whichever axis shows the stronger tremor signal
    if peak_amp_x >= peak_amp_y:
        peak_amp  = peak_amp_x
        peak_freq = float(band_freqs[np.argmax(band_x)])
        fft_full  = fft_x
    else:
        peak_amp  = peak_amp_y
        peak_freq = float(band_freqs[np.argmax(band_y)])
        fft_full  = fft_y

    band_power    = float(np.sum(fft_full[band_mask] ** 2))
    total_power   = float(np.sum(fft_full ** 2))
    relative_power = band_power / total_power if total_power > 0 else 0.0

    tremor_detected = (peak_amp >= MIN_TREMOR_AMP) and (relative_power > TREMOR_RELATIVE_POWER_THRESHOLD)
    return bool(tremor_detected), peak_amp, peak_freq


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
        self.timestamp_history = {
            "Left":  deque(maxlen=HISTORY_LENGTH),
            "Right": deque(maxlen=HISTORY_LENGTH),
        }
        # Wrist cross-validation buffer: tracks wrist (landmark 0) alongside
        # fingertip (landmark 8). Tremor must appear in BOTH to flag — filters
        # out fingertip-only noise from typing, tapping, or gesturing.
        self.wrist_history = {
            "Left":  deque(maxlen=HISTORY_LENGTH),
            "Right": deque(maxlen=HISTORY_LENGTH),
        }
        self.frames_since_seen = {"Left": 0, "Right": 0}
        self.flagged = False
        self.jitter_value = 0.0
        self.peak_freq = 0.0
        self.buffer_progress = 0  # 0-64, for warmup UI feedback
        # Sustained window: only flag if tremor appears in a majority of recent
        # analysis windows. Per-hand to avoid cross-contamination.
        # At 30fps, maxlen=300 holds ~10 seconds of history.
        self.tremor_detections = {
            "Left":  deque(maxlen=300),
            "Right": deque(maxlen=300),
        }

    def update(self, rgb_frame, timestamp_ms):
        rgb_frame.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        seen_this_frame = set()
        if result.hand_landmarks:
            h, w = rgb_frame.shape[:2]
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].category_name  # "Left" or "Right"
                seen_this_frame.add(label)
                self.frames_since_seen[label] = 0
                # Track index fingertip (landmark 8) — farther from pivot = tremor amplified
                # Wrist (landmark 0) is the most stable point; fingertip is the least stable.
                index_tip = hand_landmarks[8]
                self.position_history[label].append((int(index_tip.x * w), int(index_tip.y * h)))
                self.timestamp_history[label].append(timestamp_ms)
                # Also track wrist for cross-validation
                wrist = hand_landmarks[0]
                self.wrist_history[label].append((int(wrist.x * w), int(wrist.y * h)))
                if DEBUG:
                    print(f"[HAND] Detected {label} hand — buffer {len(self.position_history[label])}/{HISTORY_LENGTH}")
        elif DEBUG:
            print("[HAND] No hands detected this frame")

        # Increment missed-frame counters for hands not seen this frame
        for label in ("Left", "Right"):
            if label not in seen_this_frame:
                self.frames_since_seen[label] += 1
                # If hand has been gone too long, clear stale data so the FPS
                # calculation isn't corrupted by a big timestamp gap.
                if self.frames_since_seen[label] >= HAND_LOSS_RESET_FRAMES:
                    self.position_history[label].clear()
                    self.timestamp_history[label].clear()
                    self.wrist_history[label].clear()
                    self.tremor_detections[label].clear()
                    if DEBUG:
                        print(f"[RESET] Cleared {label} buffer after {HAND_LOSS_RESET_FRAMES} frames missing")

        hand_results = []
        max_buf_len = 0
        for label in ("Left", "Right"):
            buf = self.position_history[label]
            ts  = self.timestamp_history[label]
            max_buf_len = max(max_buf_len, len(buf))
            if len(buf) >= HISTORY_LENGTH:
                finger_tremor, peak_amp, peak_freq = _analyze_tremor_buffer(list(buf), list(ts))
                # Cross-validate: check if wrist also shows tremor
                wrist_buf = self.wrist_history[label]
                if len(wrist_buf) >= HISTORY_LENGTH:
                    wrist_tremor, _, _ = _analyze_tremor_buffer(list(wrist_buf), list(ts))
                else:
                    wrist_tremor = False
                # Only flag if BOTH show tremor — filters out fingertip-only noise
                # (typing, tapping, pen fidgeting)
                both_detected = finger_tremor and wrist_tremor
                hand_results.append((both_detected, peak_amp, peak_freq))
                # Update per-hand sustained window
                self.tremor_detections[label].append(1 if both_detected else 0)
                if DEBUG:
                    print(f"[FFT {label}] freq={peak_freq:.2f}Hz  amp={peak_amp:.2f}  detected={both_detected}")
            elif DEBUG:
                print(f"[FFT {label}] Buffer not full yet ({len(buf)}/{HISTORY_LENGTH}) — waiting...")

        self.buffer_progress = max_buf_len

        if hand_results:
            # jitter_value always shows the strongest peak_amp so the monitor
            # displays a live number even when no tremor is flagged.
            strongest_overall = max(hand_results, key=lambda r: r[1])
            self.jitter_value = strongest_overall[1]
            self.peak_freq = strongest_overall[2]

            # Sustained window: flag if ANY hand shows sustained tremor
            any_hand_sustained = False
            for label in ("Left", "Right"):
                if len(self.tremor_detections[label]) >= 150:
                    ratio = sum(self.tremor_detections[label]) / len(self.tremor_detections[label])
                    if ratio >= TREMOR_SUSTAINED_RATIO:
                        any_hand_sustained = True
            self.flagged = any_hand_sustained
            if DEBUG:
                if self.flagged:
                    print(f"[TREMOR] FLAGGED — strongest hand: amp={self.jitter_value:.2f}  freq={self.peak_freq:.2f}Hz")
                else:
                    print(f"[TREMOR] No tremor detected — peak amp too low or not enough spectral power (amp={self.jitter_value:.2f}  freq={self.peak_freq:.2f}Hz)")
        else:
            self.flagged = False
            self.jitter_value = 0.0
            self.peak_freq = 0.0

        return self.flagged, self.jitter_value, result
