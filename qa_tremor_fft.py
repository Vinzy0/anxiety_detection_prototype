"""
QA tests for FFT-based hand tremor detection.

Imports _analyze_tremor_buffer directly from hand_detection so any future
algorithm change is automatically reflected here -- no more drift between
test and production code.

Run from inside the anxiety_detection/ folder:
    python qa_tremor_fft.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from detection.hand_detection import (
    _analyze_tremor_buffer,
    HISTORY_LENGTH,
    TREMOR_FREQ_MIN,
    TREMOR_FREQ_MAX,
    MIN_TREMOR_AMP,
    TREMOR_RELATIVE_POWER_THRESHOLD,
)


def make_timestamps(n=HISTORY_LENGTH, fps=30.0):
    return [(i / fps) * 1000.0 for i in range(n)]


def make_positions(freq_hz, amp_px, fps=30.0, noise_px=0.5,
                   axis="both", base_drift_amp=20.0, seed=0):
    np.random.seed(seed)
    t = np.arange(HISTORY_LENGTH) / fps

    base_x = base_drift_amp * np.sin(2 * np.pi * 0.3 * t)
    base_y = base_drift_amp * np.cos(2 * np.pi * 0.2 * t)

    tremor_x = amp_px * np.sin(2 * np.pi * freq_hz * t + 0.5) if axis in ("x", "both") else np.zeros(HISTORY_LENGTH)
    tremor_y = amp_px * np.sin(2 * np.pi * freq_hz * t + 1.2) if axis in ("y", "both") else np.zeros(HISTORY_LENGTH)

    noise = np.random.normal(0, noise_px, size=(HISTORY_LENGTH, 2))
    positions = np.stack([base_x + tremor_x, base_y + tremor_y], axis=1) + noise
    return positions.astype(np.float32)


def run_tests():
    fps = 30.0
    ts  = make_timestamps(fps=fps)

    passes = 0
    fails  = 0

    def check(name, detected, amp, freq, expect_detected):
        nonlocal passes, fails
        ok = (detected == expect_detected)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         detected={detected}  amp={amp:.2f}  freq={freq:.2f} Hz"
              f"  (expected={expect_detected})")
        if ok:
            passes += 1
        else:
            fails += 1

    print(f"\nTremor band: {TREMOR_FREQ_MIN}-{TREMOR_FREQ_MAX} Hz")
    print(f"Min amplitude: {MIN_TREMOR_AMP}  Rel. power threshold: {TREMOR_RELATIVE_POWER_THRESHOLD}")
    print(f"History length: {HISTORY_LENGTH} frames\n")

    # -- Should DETECT --
    print("-- Should DETECT (true positives) --")

    pos = make_positions(freq_hz=9.0, amp_px=25.0, fps=fps, seed=1)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("9 Hz tremor, high amp, both axes", d, a, f, True)

    pos = make_positions(freq_hz=10.0, amp_px=25.0, fps=fps, seed=2)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("10 Hz tremor, high amp, both axes", d, a, f, True)

    pos = make_positions(freq_hz=8.0, amp_px=25.0, fps=fps, seed=3)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("8 Hz tremor (low edge of band), high amp", d, a, f, True)

    pos = make_positions(freq_hz=11.0, amp_px=25.0, fps=fps, seed=4)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("11 Hz tremor (high edge of band), high amp", d, a, f, True)

    pos = make_positions(freq_hz=9.0, amp_px=25.0, fps=fps, axis="x", seed=5)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("9 Hz tremor on X axis only (angle scenario)", d, a, f, True)

    pos = make_positions(freq_hz=9.0, amp_px=25.0, fps=fps, axis="y", seed=6)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("9 Hz tremor on Y axis only (angle scenario)", d, a, f, True)

    # -- Should NOT detect --
    print("\n-- Should NOT detect (true negatives) --")

    pos = make_positions(freq_hz=4.0, amp_px=25.0, fps=fps, seed=10)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("4 Hz tremor (below band)", d, a, f, False)

    pos = make_positions(freq_hz=6.0, amp_px=25.0, fps=fps, seed=11)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("6 Hz tremor (below band)", d, a, f, False)

    pos = make_positions(freq_hz=15.0, amp_px=25.0, fps=fps, seed=12)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("15 Hz tremor (above band)", d, a, f, False)

    pos = make_positions(freq_hz=9.0, amp_px=1.5, fps=fps, seed=13)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("9 Hz tremor, very low amplitude", d, a, f, False)

    np.random.seed(20)
    pos = np.random.normal(0, 2.0, size=(HISTORY_LENGTH, 2)).astype(np.float32)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("Pure noise, no rhythmic signal", d, a, f, False)

    t_arr = np.arange(HISTORY_LENGTH) / fps
    slow_x = 40.0 * np.sin(2 * np.pi * 0.5 * t_arr)
    slow_y = 40.0 * np.cos(2 * np.pi * 0.3 * t_arr)
    pos = np.stack([slow_x, slow_y], axis=1).astype(np.float32)
    d, a, f = _analyze_tremor_buffer(pos, ts)
    check("Slow deliberate arm movement (0.3-0.5 Hz)", d, a, f, False)

    # -- FPS robustness --
    print("\n-- FPS robustness --")

    ts_20 = make_timestamps(fps=20.0)
    pos = make_positions(freq_hz=9.0, amp_px=25.0, fps=20.0, seed=30)
    d, a, f = _analyze_tremor_buffer(pos, ts_20)
    check("9 Hz tremor at 20 fps", d, a, f, True)

    ts_24 = make_timestamps(fps=24.0)
    pos = make_positions(freq_hz=10.0, amp_px=25.0, fps=24.0, seed=31)
    d, a, f = _analyze_tremor_buffer(pos, ts_24)
    check("10 Hz tremor at 24 fps", d, a, f, True)

    # -- Summary --
    total = passes + fails
    print(f"\n{'='*50}")
    print(f"Results: {passes}/{total} passed")
    if fails == 0:
        print("All tests passed.")
    else:
        print(f"{fails} test(s) FAILED -- tune thresholds before running the prototype.")
    print(f"{'='*50}\n")

    return fails == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
