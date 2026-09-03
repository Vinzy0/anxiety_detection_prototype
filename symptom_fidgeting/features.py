"""Temporal movement-feature engineering for the fidgeting prototype.

Single module used by BOTH the trainer and the inference path so feature
definition and ordering can never drift between them.

Feature philosophy
------------------
Fidgeting is movement THROUGH TIME. Every feature is computed from a
30-frame window of pose landmarks. Raw coordinates are never model inputs.
Signals built per window (over pose-detected frames only):

  * RELATIVE coordinates: each joint's position minus a stable base joint
    (wrist-vs-shoulder, elbow-vs-shoulder, shoulder-vs-nose, hip-vs-nose)
    -> removes camera framing and static posture offsets.
  * Motion of those relative coordinates over time = the fidget signal.

Per signal we summarize: total displacement, mean/peak velocity magnitude,
velocity spread, direction (sign) changes, position spread. All distance /
velocity quantities are divided by the window's shoulder width so the model
does not learn camera resolution, distance, or body size (units:
shoulder-widths per frame).

Quality features: pose detection rate and mean wrist confidence.
"""

import numpy as np

from . import paths
from .pose_utils import (REFERENCE_PAIRS, SCALE_LANDMARKS, DETECTED_COL,
                         CONF_COLS, _col)

# Which per-signal statistics are computed, in order. Each relative-coordinate
# axis yields one of these blocks.
SIGNAL_STATS = ("vel_total", "vel_mean", "vel_std", "vel_max",
                "dir_changes", "pos_mean", "pos_std")

GLOBAL_FEATURES = ("detection_rate", "wrist_conf_mean")


def _stat(values):
    """(mean, std) of a finite series; (0, 0) for empty/broken input."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0 or not np.all(np.isfinite(a)):
        return 0.0, 0.0
    return float(np.mean(a)), float(np.std(a))


def _velocities(x):
    """Frame-to-frame velocity of a 1-D signal (first frame -> 0)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return np.zeros(x.size if x.ndim == 1 else 0, dtype=np.float64)
    return np.concatenate([[0.0], np.diff(x)])


def _signal_features(signal, scale):
    """7 stats for one relative-coordinate signal, scale-normalized."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0 or not np.all(np.isfinite(signal)):
        return [0.0] * len(SIGNAL_STATS)
    scale = scale if scale and scale > 0 else 1.0
    v = _velocities(signal)
    mag = np.abs(v)
    # Direction changes = sign flips of the velocity (zeros skipped).
    nz = v[v != 0.0]
    n_changes = int(np.count_nonzero(np.diff(np.sign(nz))) if nz.size > 1
                    else 0)
    pos_mean, pos_std = _stat(signal)
    return [float(np.sum(mag)) / scale,
            float(np.mean(mag)) / scale,
            float(np.std(mag)) / scale,
            float(np.max(mag)) / scale if mag.size else 0.0,
            float(n_changes),
            pos_mean / scale,
            pos_std / scale]


def feature_names():
    """Canonical ordered feature names (must match window_to_features)."""
    names = []
    for joint, base in REFERENCE_PAIRS:
        for axis in ("x", "y"):
            stem = f"rel_{joint}_vs_{base}_{axis}"
            names += [f"{stem}_{s}" for s in SIGNAL_STATS]
    for side in ("l", "r"):
        names += [f"idxwrist_{side}_mean", f"idxwrist_{side}_std"]
    names += list(GLOBAL_FEATURES)
    return names


def _scale_from_window(df):
    """Shoulder width (normalized units) from detected frames; 1.0 if N/A."""
    for colx in (_col(SCALE_LANDMARKS[0], "x"), _col(SCALE_LANDMARKS[1], "x")):
        if colx not in df.columns:
            return 1.0
    lx = df[colx].to_numpy()
    rx = df[_col(SCALE_LANDMARKS[1], "x")].to_numpy()
    width = float(np.nanmedian(np.abs(lx - rx)))
    return width if width > 1e-4 else 1.0


def _rel_series(df, detected_mask, joint, base):
    """Relative-coordinate series over detected frames for one axis."""
    sx = _col(joint, "x"); sy = _col(joint, "y")
    bx = _col(base, "x"); by = _col(base, "y")
    if sx not in df.columns or bx not in df.columns:
        return None, None
    jx = df.loc[detected_mask, sx].to_numpy()
    jy = df.loc[detected_mask, sy].to_numpy()
    rxx = df.loc[detected_mask, bx].to_numpy()
    ryy = df.loc[detected_mask, by].to_numpy()
    return jx - rxx, jy - ryy


def window_to_features(df, scale=None, min_detection_rate=None,
                       min_detected_frames=None):
    """Convert one landmark window (DataFrame row range) into features.

    Returns (feature_vector, det_rate), or (None, det_rate) when the window
    is too poorly detected to be usable. Feature order == feature_names().
    """
    min_detection_rate = (paths.MIN_DETECTION_RATE if min_detection_rate is
                          None else min_detection_rate)
    min_detected_frames = (paths.MIN_DETECTED_FRAMES if min_detected_frames is
                           None else min_detected_frames)

    if DETECTED_COL not in df.columns:
        return None, 0.0
    det = df[DETECTED_COL].astype(float).to_numpy()
    detected_mask = det > 0.5
    det_rate = float(det.mean()) if det.size else 0.0
    n_det = int(detected_mask.sum())

    if det_rate < min_detection_rate or n_det < min_detected_frames:
        return None, det_rate

    if scale is None:
        scale = _scale_from_window(df)

    feats = []
    for joint, base in REFERENCE_PAIRS:
        sx, sy = _rel_series(df, detected_mask, joint, base)
        feats += _signal_features(sx, scale)
        feats += _signal_features(sy, scale)

    # Finger-fist distance per side (wrist-index proximity change).
    for side in ("l", "r"):
        ix = _col(f"{side}_index", "x"); wx = _col(f"{side}_wrist", "x")
        iy = _col(f"{side}_index", "y"); wy = _col(f"{side}_wrist", "y")
        if ix in df.columns and wx in df.columns:
            d = np.hypot(df.loc[detected_mask, ix] - df.loc[detected_mask, wx],
                         df.loc[detected_mask, iy] - df.loc[detected_mask, wy])
            m, s = _stat(d)
            feats += [m / scale if scale else m, s / scale if scale else s]
        else:
            feats += [0.0, 0.0]

    # Quality features.
    confs = [df[c].astype(float).to_numpy() for c in CONF_COLS
             if c in df.columns]
    conf_mean = float(np.mean(confs)) if confs else 0.0
    feats += [det_rate, conf_mean]
    return np.asarray(feats, dtype=np.float64), det_rate
