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

TREMOR_FREQ_MIN = 4.0
TREMOR_FREQ_MAX = 12.0
MIN_TREMOR_AMP = 10.0
# TODO: threshold was calibrated against single-bin amplitude ratio (old formula).
# Now uses band_power/total_power (magnitude²). Needs empirical re-tuning — 0.35
# may be too high or too low under the new scale. Run qa_tremor_fft.py after tuning.
TREMOR_RELATIVE_POWER_THRESHOLD = 0.35

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

    Parameters
    ----------
    positions : list or array of shape (N, 2)
        Raw (x, y) wrist pixel coordinates.
    timestamps : list or array of shape (N,)
        Frame timestamps in milliseconds.

    Returns
    -------
    tuple (tremor_detected, peak_amp, peak_freq)
        tremor_detected : bool
        peak_amp        : float — FFT magnitude of the dominant peak
        peak_freq       : float — frequency (Hz) of the dominant peak
    """
    positions = np.array(positions)  # shape (N, 2)
    deltas = np.diff(positions, axis=0)
    displacement = np.linalg.norm(deltas, axis=1)  # shape (N-1,)
    displacement = displacement - np.mean(displacement)  # remove DC offset
    windowed = displacement * np.hanning(len(displacement))  # reduce spectral leakage

    # Compute real FPS from elapsed time between frames — webcam FPS can vary
    # due to processing load, so we measure it directly instead of assuming 30.
    elapsed_s = (timestamps[-1] - timestamps[0]) / 1000.0
    real_fps = len(displacement) / elapsed_s if elapsed_s > 0 else 30.0
    # N-1 displacements over N-1 timestamp intervals → correct FPS, not off-by-one

    fft_mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / real_fps)

    band_mask = (freqs >= TREMOR_FREQ_MIN) & (freqs <= TREMOR_FREQ_MAX)
    band_mags = fft_mag[band_mask]
    band_freqs = freqs[band_mask]

    if len(band_mags) == 0:
        return False, 0.0, 0.0

    peak_idx = np.argmax(band_mags)
    peak_freq = band_freqs[peak_idx]
    peak_amp = band_mags[peak_idx]
    band_power = np.sum(band_mags ** 2)
    total_power = np.sum(fft_mag ** 2)
    relative_power = band_power / total_power if total_power > 0 else 0
    tremor_detected = (peak_amp >= MIN_TREMOR_AMP) and (relative_power > TREMOR_RELATIVE_POWER_THRESHOLD)
    return bool(tremor_detected), float(peak_amp), float(peak_freq)


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
        self.frames_since_seen = {"Left": 0, "Right": 0}
        self.flagged = False
        self.jitter_value = 0.0
        self.peak_freq = 0.0
        self.buffer_progress = 0  # 0-64, for warmup UI feedback

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
                wrist = hand_landmarks[0]
                self.position_history[label].append((int(wrist.x * w), int(wrist.y * h)))
                self.timestamp_history[label].append(timestamp_ms)
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
                    if DEBUG:
                        print(f"[RESET] Cleared {label} buffer after {HAND_LOSS_RESET_FRAMES} frames missing")

        hand_results = []
        max_buf_len = 0
        for label in ("Left", "Right"):
            buf = self.position_history[label]
            ts  = self.timestamp_history[label]
            max_buf_len = max(max_buf_len, len(buf))
            if len(buf) >= HISTORY_LENGTH:
                tremor_detected, peak_amp, peak_freq = _analyze_tremor_buffer(list(buf), list(ts))
                hand_results.append((tremor_detected, peak_amp, peak_freq))
                if DEBUG:
                    print(f"[FFT {label}] freq={peak_freq:.2f}Hz  amp={peak_amp:.2f}  detected={tremor_detected}")
            elif DEBUG:
                print(f"[FFT {label}] Buffer not full yet ({len(buf)}/{HISTORY_LENGTH}) — waiting...")

        self.buffer_progress = max_buf_len

        if hand_results:
            # jitter_value always shows the strongest peak_amp so the monitor
            # displays a live number even when no tremor is flagged.
            strongest_overall = max(hand_results, key=lambda r: r[1])
            self.jitter_value = strongest_overall[1]
            self.peak_freq = strongest_overall[2]

            tremor_flags = [r[0] for r in hand_results]
            self.flagged = any(tremor_flags)
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
