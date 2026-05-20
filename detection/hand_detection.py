import os
import numpy as np
from collections import deque
import mediapipe as mp

HISTORY_LENGTH = 64
JITTER_THRESHOLD = 8.0  # kept for backward compatibility, superseded by MIN_TREMOR_AMP

# Clinical tremor range is 8–12 Hz; 7.5 gives a half-bin buffer against FFT quantization.
TREMOR_FREQ_MIN = 7.5
TREMOR_FREQ_MAX = 12.0
MIN_TREMOR_AMP = 50.0
TREMOR_RELATIVE_POWER_THRESHOLD = 0.35  # tremor band must be >35% of total signal power

# Tremor must appear in ≥50% of the last ~10 s of windows to avoid false positives from bursts.
TREMOR_SUSTAINED_RATIO = 0.5

# Clear a hand's buffer after this many consecutive missed frames so stale timestamps
# don't corrupt the FPS calculation when the hand reappears.+__
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
    Run the FFT tremor detection pipeline on a single hand's position buffer.
    Returns (tremor_detected, peak_amp, peak_freq).

    Uses raw position rather than frame-to-frame displacement: a constant-speed
    sweep has a flat displacement signal whose harmonics bleed into the tremor
    band, whereas its position signal is low-frequency with no 8–12 Hz energy.
    """
    # ── Step 1: collect positions ──────────────────────────────────────────────
    positions = np.array(positions, dtype=np.float64)  # shape (N, 2)
    N = len(positions)
    x = positions[:, 0]
    y = positions[:, 1]

    # ── Step 2: detrend & remove DC offset ────────────────────────────────────
    # Fit and subtract a straight line from each axis so slow drifts and sweeps
    # don't mask the tremor oscillation.
    t_idx = np.arange(N, dtype=np.float64)
    x -= np.polyval(np.polyfit(t_idx, x, 1), t_idx)
    y -= np.polyval(np.polyfit(t_idx, y, 1), t_idx)
    x -= np.mean(x)
    y -= np.mean(y)

    # ── Step 3: apply Hanning window ──────────────────────────────────────────
    # Tapers the signal at both ends so edge discontinuities don't smear energy
    # across unrelated frequency bins (spectral leakage).
    window = np.hanning(N)
    x_w = x * window
    y_w = y * window

    # ── Step 4: compute real FPS & run FFT ────────────────────────────────────
    # Derive FPS from actual timestamps; webcam delivery rate varies under load.
    # FFT each axis independently so tremor is caught even if the hand is
    # seen edge-on and only one axis carries the motion.
    elapsed_s = (timestamps[-1] - timestamps[0]) / 1000.0
    real_fps = (N - 1) / elapsed_s if elapsed_s > 0 else 30.0

    freqs = np.fft.rfftfreq(N, d=1.0 / real_fps)
    fft_x = np.abs(np.fft.rfft(x_w))
    fft_y = np.abs(np.fft.rfft(y_w))

    # ── Step 5: isolate the tremor band (7.5–12 Hz) ───────────────────────────
    band_mask = (freqs >= TREMOR_FREQ_MIN) & (freqs <= TREMOR_FREQ_MAX)
    if not np.any(band_mask):
        return False, 0.0, 0.0

    band_freqs = freqs[band_mask]
    band_x = fft_x[band_mask]
    band_y = fft_y[band_mask]

    # Use whichever axis shows the stronger peak.
    peak_amp_x = float(np.max(band_x))
    peak_amp_y = float(np.max(band_y))
    if peak_amp_x >= peak_amp_y:
        peak_amp  = peak_amp_x
        peak_freq = float(band_freqs[np.argmax(band_x)])
        fft_full  = fft_x
    else:
        peak_amp  = peak_amp_y
        peak_freq = float(band_freqs[np.argmax(band_y)])
        fft_full  = fft_y

    # ── Step 6: two-gate tremor check ─────────────────────────────────────────
    # Gate 1 — absolute amplitude: peak must be physically large enough.
    # Gate 2 — relative power: tremor band must dominate the spectrum (>35%),
    #           so broadband noise with a coincidental in-band spike is rejected.
    band_power     = float(np.sum(fft_full[band_mask] ** 2))
    total_power    = float(np.sum(fft_full ** 2))
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
        # Step 7 (cross-validation): wrist (landmark 0) must also show tremor
        # alongside the fingertip (landmark 8) to rule out typing/tapping noise.
        self.wrist_history = {
            "Left":  deque(maxlen=HISTORY_LENGTH),
            "Right": deque(maxlen=HISTORY_LENGTH),
        }
        self.frames_since_seen = {"Left": 0, "Right": 0}
        self.flagged = False
        self.jitter_value = 0.0
        self.peak_freq = 0.0
        self.buffer_progress = 0  # 0–64, shown in warmup UI
        # Step 8 (sustained window): maxlen=300 ≈ 10 s at 30 fps, per-hand.
        self.tremor_detections = {
            "Left":  deque(maxlen=300),
            "Right": deque(maxlen=300),
        }

    def update(self, rgb_frame, timestamp_ms):
        rgb_frame.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # ── Step 1: collect positions ──────────────────────────────────────────
        # Fingertip (landmark 8) amplifies tremor most; wrist (landmark 0) is
        # used for cross-validation in step 7.
        seen_this_frame = set()
        if result.hand_landmarks:
            h, w = rgb_frame.shape[:2]
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].category_name  # "Left" or "Right"
                seen_this_frame.add(label)
                self.frames_since_seen[label] = 0
                index_tip = hand_landmarks[8]
                self.position_history[label].append((int(index_tip.x * w), int(index_tip.y * h)))
                self.timestamp_history[label].append(timestamp_ms)
                wrist = hand_landmarks[0]
                self.wrist_history[label].append((int(wrist.x * w), int(wrist.y * h)))
                if DEBUG:
                    print(f"[HAND] Detected {label} hand — buffer {len(self.position_history[label])}/{HISTORY_LENGTH}")
        elif DEBUG:
            print("[HAND] No hands detected this frame")

        # Clear buffers for hands that have been absent too long so a large
        # timestamp gap doesn't corrupt the FPS calculation when they return.
        for label in ("Left", "Right"):
            if label not in seen_this_frame:
                self.frames_since_seen[label] += 1
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
                # Steps 2–6: run the FFT pipeline on the fingertip buffer.
                finger_tremor, peak_amp, peak_freq = _analyze_tremor_buffer(list(buf), list(ts))

                # ── Step 7: cross-validate with wrist ─────────────────────────
                # Typing and tapping show up only at the fingertip; true tremor
                # appears at the wrist too.
                wrist_buf = self.wrist_history[label]
                if len(wrist_buf) >= HISTORY_LENGTH:
                    wrist_tremor, _, _ = _analyze_tremor_buffer(list(wrist_buf), list(ts))
                else:
                    wrist_tremor = False
                both_detected = finger_tremor and wrist_tremor
                hand_results.append((both_detected, peak_amp, peak_freq))

                # ── Step 8: update sustained window ───────────────────────────
                self.tremor_detections[label].append(1 if both_detected else 0)
                if DEBUG:
                    print(f"[FFT {label}] freq={peak_freq:.2f}Hz  amp={peak_amp:.2f}  detected={both_detected}")
            elif DEBUG:
                print(f"[FFT {label}] Buffer not full yet ({len(buf)}/{HISTORY_LENGTH}) — waiting...")

        self.buffer_progress = max_buf_len

        if hand_results:
            # Always expose the strongest peak_amp so the UI has a live value
            # even when no tremor is flagged.
            strongest_overall = max(hand_results, key=lambda r: r[1])
            self.jitter_value = strongest_overall[1]
            self.peak_freq = strongest_overall[2]

            # ── Step 8: sustained window ──────────────────────────────────────
            # Require ≥150 windows (~5 s) before deciding, then flag if ≥50%
            # of those windows detected tremor on at least one hand.
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
                    print(f"[TREMOR] No tremor — amp={self.jitter_value:.2f}  freq={self.peak_freq:.2f}Hz")
        else:
            self.flagged = False
            self.jitter_value = 0.0
            self.peak_freq = 0.0

        return self.flagged, self.jitter_value, result
