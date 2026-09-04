"""Regression test for shoulder-width normalization (v1 bug).

v1 `_scale_from_window` bound the loop variable `colx` to the LAST iterated
column name (right-shoulder x) before subtracting, so it computed
right_x - right_x = 0 -> fell back to a spurious 1.0. Every window scaled
by 1.0; the model never saw real body size. This test must fail if that
regression ever returns:

  * real-data windows must produce a realistic scale (~0.1-0.4), never 1.0
  * a synthetic window with known shoulder separation must return it

Run:  python -m unittest symptom_fidgeting.tests.test_scale -v
"""
import os
import unittest

import numpy as np
import pandas as pd

from symptom_fidgeting import paths
from symptom_fidgeting.pose_utils import (CONF_PREFIX, DETECTED_COL,
                                          SCALE_LANDMARKS, _col)
from symptom_fidgeting.features import _scale_from_window

REAL_RANGE = (0.05, 0.9)  # plausible normalized shoulder width, any person


def _synthetic_window(n=30, lx=0.35, rx=0.60, width_y=0.0, n_detected=30,
                      conf=0.9, nan_frames=0, seed=0):
    """Window with fixed true shoulder separation ~hypot(rx-lx, width_y)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        detected = i < n_detected
        rows.append({DETECTED_COL: 1.0 if detected else 0.0})
        for j, (x, y) in {
            "l_shoulder": (lx, 0.5 + width_y / 2),
            "r_shoulder": (rx, 0.5 - width_y / 2),
        }.items():
            rows[-1][_col(j, "x")] = x + (rng.normal(0, 1e-4) if detected else np.nan)
            rows[-1][_col(j, "y")] = y + (rng.normal(0, 1e-4) if detected else np.nan)
            rows[-1][CONF_PREFIX + j] = conf if detected else np.nan
        for c in ("lm_l_wrist_x", "lm_r_wrist_x"):
            rows[-1][c] = 0.5 if detected else np.nan
    df = pd.DataFrame(rows)
    if nan_frames:
        # extra low-confidence/NaN frames that must NOT move the median
        for i in range(min(nan_frames, n_detected)):
            df.loc[i, _col("l_shoulder", "x")] = np.nan
            df.loc[i, CONF_PREFIX + "l_shoulder"] = 0.05
    return df


class ScaleRegressionTests(unittest.TestCase):

    def test_synthetic_known_width(self):
        df = _synthetic_window(lx=0.35, rx=0.60)  # true width 0.25
        s = _scale_from_window(df)
        self.assertGreater(s, 0.0)
        self.assertNotAlmostEqual(s, 1.0)
        self.assertAlmostEqual(s, 0.25, delta=0.01)

    def test_synthetic_uses_y_too(self):
        # 0.20 horizontal + 0.15 vertical separation -> sqrt(0.04+0.0225)
        df = _synthetic_window(lx=0.40, rx=0.60, width_y=0.15)
        s = _scale_from_window(df)
        self.assertAlmostEqual(s, 0.25, delta=0.01)

    def test_synthetic_missing_frames_still_median(self):
        df = _synthetic_window(n=40, n_detected=25, nan_frames=6, lx=0.35,
                               rx=0.60)
        s = _scale_from_window(df)
        self.assertAlmostEqual(s, 0.25, delta=0.01)

    def test_all_low_confidence_returns_zero(self):
        df = _synthetic_window(conf=0.05)
        self.assertEqual(_scale_from_window(df), 0.0)

    def test_no_detected_frames_returns_zero(self):
        df = _synthetic_window(n_detected=0)
        self.assertEqual(_scale_from_window(df), 0.0)

    def test_zero_width_returns_zero_not_one(self):
        df = _synthetic_window(lx=0.5, rx=0.5)
        # nudge noise off: exactly coincident shoulders -> truly zero width
        for j in ("l_shoulder", "r_shoulder"):
            df[_col(j, "x")] = 0.5
            df[_col(j, "y")] = 0.5
        self.assertEqual(_scale_from_window(df), 0.0)

    def test_real_extracted_mas_windows(self):
        """Real extracted windows: scale is realistic and never spuriously 1.0."""
        csvs = sorted(f for f in os.listdir(paths.EXTRACTED_DIR)
                      if f.endswith(".csv"))[:6]
        self.assertTrue(csvs, "no extracted CSVs found")
        scales = []
        for f in csvs:
            df = pd.read_csv(os.path.join(paths.EXTRACTED_DIR, f))
            n = len(df)
            for s in range(0, max(0, n - paths.WINDOW_FRAMES) + 1,
                           paths.WINDOW_STEP):
                w = df.iloc[s:s + paths.WINDOW_FRAMES]
                sc = _scale_from_window(w)
                if sc > 0:
                    scales.append(sc)
        self.assertTrue(len(scales) > 20, f"too few usable windows: {len(scales)}")
        scales = np.array(scales)
        self.assertTrue(np.all(scales > REAL_RANGE[0]))
        self.assertTrue(np.all(scales < REAL_RANGE[1]))
        self.assertTrue(np.all(scales != 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
