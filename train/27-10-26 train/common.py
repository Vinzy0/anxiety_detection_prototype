import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(os.environ.get("THESIS_ROOT", r"F:\Thesis\videos\New folder\Thesis Data Gathering"))
INPUT_CSV = ROOT / "symptom_frames.csv.gz"
OUT = ROOT / "v6"
FEATURES_PKL, SEQ_NPY, INFO_JSON = OUT / "features.pkl", OUT / "sequences.npy", OUT / "feature_info.json"

SYMPTOMS = ["fidgeting", "hand_tremor", "knee_tremor", "avoidant_gaze", "facial_tension", "heavy_breathing", "fear"]
CLASSES = ["control"] + SYMPTOMS
WINDOW = dict(trim_s=1.0, win_s=4.0, step_s=2.0, fs=30.0, min_valid_frac=0.5)
FS = WINDOW["fs"]
BANDS = {"breath": (0.15, 0.8), "move": (0.8, 3.0), "tremor": (3.0, 9.0)}
STATS = ["mean", "std", "range", "vel_mean", "vel_p95", "log_power", "dom_freq", "band_breath", "band_move", "band_tremor"]
PSTATS = ["std", "vel_mean", "band_breath", "band_move", "band_tremor"]
HAND_POINTS = [0, 4, 8, 12, 20]
POSE_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
FACE_POINTS = [1, 13, 14, 33, 61, 105, 133, 145, 159, 263, 291, 334, 362, 374, 386, 468, 473]
FLAGS = ["face_detected", "pose_detected", "left_hand_detected", "right_hand_detected"]
RAW_META = ["participant_id", "angle", "symptom", "video_id", "frame", "timestamp_ms", "fps"]
LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", random_state=42, verbose=-1)
N_FOLDS, SEED, THRESHOLD, OUTLIER_FRAC, MIN_OUTLIER_WINDOWS, POSE_VIS_MIN = 5, 42, 0.5, 0.5, 3, 0.3

PART_COLS = {
    "face": [f"face_{i}_{a}" for i in range(478) for a in "xy"],
    "pose": [f"pose_{i}_{a}" for i in range(33) for a in "xy"],
    "lh": [f"left_hand_{i}_{a}" for i in range(21) for a in "xy"],
    "rh": [f"right_hand_{i}_{a}" for i in range(21) for a in "xy"],
}
VIS_COLS = [f"pose_{i}_visibility" for i in range(33)]
_BROW_L, _BROW_R = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107], [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
_EYE_L = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
_EYE_R = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
_LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308,
         310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def _ch(ids):
    return np.array([c for i in ids for c in (2 * i, 2 * i + 1)])


GROUPS = [("face_all", "face", _ch(range(478))), ("face_brow_l", "face", _ch(_BROW_L)), ("face_brow_r", "face", _ch(_BROW_R)),
          ("face_eye_l", "face", _ch(_EYE_L)), ("face_eye_r", "face", _ch(_EYE_R)), ("face_iris", "face", _ch(range(468, 478))),
          ("face_lips", "face", _ch(_LIPS)), ("face_nose", "face", _ch([1, 2, 4, 5, 6, 19, 94, 98, 168, 195, 197, 326, 327])),
          ("face_oval", "face", _ch(_OVAL)), ("pose_all", "pose", _ch(range(33))), ("pose_head", "pose", _ch(range(11))),
          ("pose_arm_l", "pose", _ch([11, 13, 15, 17, 19, 21])), ("pose_arm_r", "pose", _ch([12, 14, 16, 18, 20, 22])),
          ("pose_torso", "pose", _ch([11, 12, 23, 24])), ("pose_leg_l", "pose", _ch([23, 25, 27, 29, 31])),
          ("pose_leg_r", "pose", _ch([24, 26, 28, 30, 32]))]
for _side in ("lh", "rh"):
    GROUPS += [(f"{_side}_all", _side, _ch(range(21))), (f"{_side}_thumb", _side, _ch(range(1, 5))),
               (f"{_side}_index", _side, _ch(range(5, 9))), (f"{_side}_middle", _side, _ch(range(9, 13))),
               (f"{_side}_ring", _side, _ch(range(13, 17))), (f"{_side}_pinky", _side, _ch(range(17, 21))),
               (f"{_side}_palm", _side, _ch([0, 5, 9, 13, 17]))]
GROUP_NAMES = [g[0] for g in GROUPS]
POOLED_NAMES = [f"p_{g}_{agg}_{st}" for g in GROUP_NAMES for agg in ("mean", "max") for st in PSTATS]
_HANN = {}


def landmark_cols():
    c = []
    for side in ("left_hand", "right_hand"):
        c += [f"{side}_{p}_{a}" for p in HAND_POINTS for a in "xy"]
    c += [f"pose_{p}_{a}" for p in POSE_POINTS for a in "xy"]
    c += [f"face_{p}_{a}" for p in FACE_POINTS for a in "xy"]
    return c


def raw_columns(bs, fm):
    return list(dict.fromkeys(RAW_META + FLAGS + sum(PART_COLS.values(), []) + VIS_COLS + bs + fm))


def videos(path, cols, chunk=10000):
    buf = None
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunk, low_memory=False,
                          dtype={"video_id": str, "participant_id": str, "angle": str, "symptom": str}):
        if buf is not None:
            ch = pd.concat([buf, ch], ignore_index=True)
        diff = np.flatnonzero((ch["video_id"] != ch["video_id"].iloc[-1]).to_numpy())
        if len(diff):
            cut = diff[-1] + 1
            buf = ch.iloc[cut:]
            for _, v in ch.iloc[:cut].groupby("video_id", sort=False):
                yield v.reset_index(drop=True)
        else:
            buf = ch
    if buf is not None and len(buf):
        for _, v in buf.groupby("video_id", sort=False):
            yield v.reset_index(drop=True)


def _dist(c, pre, i, j):
    return np.hypot(c[f"{pre}_{i}_x"] - c[f"{pre}_{j}_x"], c[f"{pre}_{i}_y"] - c[f"{pre}_{j}_y"])


def _proj(c, p, a, b):
    ax, ay = c[f"face_{a}_x"], c[f"face_{a}_y"]
    dx, dy = c[f"face_{b}_x"] - ax, c[f"face_{b}_y"] - ay
    return ((c[f"face_{p}_x"] - ax) * dx + (c[f"face_{p}_y"] - ay) * dy) / (dx * dx + dy * dy)


def compute_signals(c):
    s = {}
    with np.errstate(all="ignore"):
        for side in ("left_hand", "right_hand"):
            for p in HAND_POINTS:
                for a in "xy":
                    s[f"{side}_{p}_{a}"] = c[f"{side}_{p}_{a}"]
            s[f"{side}_openness"] = _dist(c, side, 0, 12)
        for p in POSE_POINTS:
            for a in "xy":
                s[f"pose_{p}_{a}"] = c[f"pose_{p}_{a}"]
        sh = (c["pose_11_y"] + c["pose_12_y"]) / 2
        s["pose_shoulder_mid_y"] = sh
        s["pose_shoulder_width"] = _dist(c, "pose", 11, 12)
        s["pose_torso_len"] = (c["pose_23_y"] + c["pose_24_y"]) / 2 - sh
        iod = _dist(c, "face", 33, 263)
        ex, ey = (c["face_33_x"] + c["face_263_x"]) / 2, (c["face_33_y"] + c["face_263_y"]) / 2
        s["face_nose_x"], s["face_nose_y"] = c["face_1_x"] / iod, c["face_1_y"] / iod
        s["face_gaze_h_468"], s["face_gaze_h_473"] = _proj(c, 468, 33, 133), _proj(c, 473, 362, 263)
        s["face_gaze_v_468"], s["face_gaze_v_473"] = _proj(c, 468, 159, 145), _proj(c, 473, 386, 374)
        s["face_head_yaw"], s["face_head_pitch"] = (c["face_1_x"] - ex) / iod, (c["face_1_y"] - ey) / iod
        s["face_mouth_open"], s["face_mouth_width"] = _dist(c, "face", 13, 14) / iod, _dist(c, "face", 61, 291) / iod
        s["face_brow_105"], s["face_brow_334"] = _dist(c, "face", 105, 159) / iod, _dist(c, "face", 334, 386) / iod
    names = list(s)
    S = np.column_stack([np.asarray(v, np.float64) for v in s.values()])
    S[~np.isfinite(S)] = np.nan
    return names, S


def signal_stats(G):
    T = G.shape[-1]
    invalid = np.isnan(G).any(-1)
    mean, std = G.mean(-1), G.std(-1)
    p5, p95 = np.percentile(G, [5, 95], axis=-1)
    vel = np.abs(np.diff(G, axis=-1)) * FS
    P = np.abs(np.fft.rfft((G - mean[..., None]) * np.hanning(T), axis=-1)) ** 2
    f = np.fft.rfftfreq(T, 1.0 / FS)
    P, f = P[..., 1:], f[1:]
    tot = P.sum(-1)
    dom = np.where(tot > 0, f[np.argmax(np.nan_to_num(P, nan=-1.0), axis=-1)], np.nan)
    bands = [np.where(tot > 0, P[..., (f >= lo) & ((f <= hi) if n == "tremor" else (f < hi))].sum(-1) / tot, np.nan)
             for n, (lo, hi) in BANDS.items()]
    out = np.stack([mean, std, p95 - p5, vel.mean(-1), np.percentile(vel, 95, axis=-1), np.log10(tot + 1e-12), dom] + bands, -1)
    out[invalid] = np.nan
    return out


def interp_part(tt, seg, t0, off):
    ok = np.isfinite(seg[:, 0]) if len(seg) else np.zeros(0, bool)
    if len(tt) < 2 or ok.sum() < max(2, WINDOW["min_valid_frac"] * len(tt)):
        return None
    tv, Lv = tt[ok], seg[ok].astype(np.float64)
    q = t0 + off
    idx = np.searchsorted(tv, q)
    right, left = np.clip(idx, 0, len(tv) - 1), np.clip(idx - 1, 0, len(tv) - 1)
    dt = tv[right] - tv[left]
    w = np.clip(np.where(dt > 0, (q - tv[left]) / np.where(dt > 0, dt, 1.0), 0.0), 0, 1)
    return Lv[left] * (1 - w)[:, None] + Lv[right] * w[:, None]


def channel_stats(W, n_channels):
    out = np.full((len(PSTATS), n_channels), np.nan)
    if W is None:
        return out
    valid = np.isfinite(W).all(0)
    if not valid.any():
        return out
    X, T = W[:, valid], W.shape[0]
    if T not in _HANN:
        f = np.fft.rfftfreq(T, 1.0 / FS)[1:]
        _HANN[T] = (np.hanning(T)[:, None], f, [(f >= 0.15) & (f < 0.8), (f >= 0.8) & (f < 3.0), (f >= 3.0) & (f <= 9.0)])
    hann, f, masks = _HANN[T]
    P = np.abs(np.fft.rfft((X - X.mean(0)) * hann, axis=0))[1:] ** 2
    tot = P.sum(0)
    fr = [np.where(tot > 0, P[m].sum(0) / np.where(tot > 0, tot, 1), np.nan) for m in masks]
    out[:, valid] = np.stack([X.std(0), np.abs(np.diff(X, axis=0)).mean(0) * FS] + fr)
    return out


def pooled_window(raw, lo, hi, t0, off):
    T, tt = len(off), raw["t"][lo:hi]
    W = {p: interp_part(tt, raw[p][lo:hi], t0, off) for p in PART_COLS}
    if W["pose"] is not None:
        vis = np.nanmean(raw["vis"][lo:hi], axis=0)
        W["pose"][:, np.repeat(~(vis >= POSE_VIS_MIN), 2)] = np.nan
    cs = {p: channel_stats(W[p], len(PART_COLS[p])) for p in PART_COLS}
    feats, energy = [], np.full((len(GROUPS), T), np.nan)
    for gi, (_, part, idx) in enumerate(GROUPS):
        s = cs[part][:, idx]
        feats += [np.nanmean(s, 1), np.nanmax(s, 1)]
        if W[part] is not None:
            X = W[part][:, idx]
            X = X[:, np.isfinite(X).all(0)]
            if X.shape[1]:
                e = np.abs(np.diff(X, axis=0)).mean(1) * FS
                energy[gi] = np.r_[e[0], e]
    return np.concatenate(feats), energy


def video_raw(v, bs, fm, t):
    arr = lambda cols: np.column_stack([pd.to_numeric(v[c], errors="coerce").to_numpy(np.float32) for c in cols])
    lm = {c: pd.to_numeric(v[c], errors="coerce").to_numpy(np.float64) for c in landmark_cols()}
    names, S = compute_signals(lm)
    raw = dict(t=t, S=S, signal_names=names, B=arr(bs), M=arr(fm), F=arr(FLAGS), vis=arr(VIS_COLS))
    raw.update({p: arr(cols) for p, cols in PART_COLS.items()})
    return raw


def default_starts(t):
    return np.arange(t[0] + WINDOW["trim_s"], t[-1] - WINDOW["trim_s"] - WINDOW["win_s"] + 1e-9, WINDOW["step_s"])


def window_all(raw, starts):
    win = WINDOW["win_s"]
    T = int(round(win * FS))
    off = np.arange(T) / FS
    t, S, B, M, F = raw["t"], raw["S"], raw["B"], raw["M"], raw["F"]
    G = np.full((len(starts), S.shape[1], T), np.nan)
    E = np.full((len(starts), len(GROUPS), T), np.nan)
    extra_dim = B.shape[1] * 3 + M.shape[1] * 2 + F.shape[1]
    extra, pooled = [], []
    for w, t0 in enumerate(starts):
        lo, hi = np.searchsorted(t, [t0, t0 + win])
        n, tt, seg = hi - lo, t[lo:hi], S[lo:hi]
        pf, E[w] = pooled_window(raw, lo, hi, t0, off)
        pooled.append(pf)
        if n < 2:
            extra.append(np.full(extra_dim, np.nan))
            continue
        valid = ~np.isnan(seg)
        nv = valid.sum(0)
        for j in np.flatnonzero((nv >= WINDOW["min_valid_frac"] * n) & (nv >= 2)):
            G[w, j] = np.interp(t0 + off, tt[valid[:, j]], seg[valid[:, j], j])
        b, m, fl = B[lo:hi], M[lo:hi], F[lo:hi]
        extra.append(np.concatenate([np.stack([np.nanmean(b, 0), np.nanstd(b, 0), np.nanmax(b, 0)], 1).ravel(),
                                     np.stack([np.nanmean(m, 0), np.nanstd(m, 0)], 1).ravel(), np.nanmean(fl, 0)]))
    v4 = np.hstack([signal_stats(G).reshape(len(starts), -1), np.vstack(extra)])
    return v4, np.vstack(pooled), G, E


def v4_names(sig, bs, fm):
    return [f"f_{s}_{st}" for s in sig for st in STATS] + [f"f_{b}_{st}" for b in bs for st in ("mean", "std", "max")] + \
           [f"f_{m}_{st}" for m in fm for st in ("mean", "std")] + [f"f_{fl}_rate" for fl in FLAGS]


FEATURE_SETS = ["v4", "landmarks", "both"]


def load(features="both"):
    if not FEATURES_PKL.is_file():
        raise FileNotFoundError(f"{FEATURES_PKL} missing. Run build_features_v6.py first.")
    info = json.loads(INFO_JSON.read_text())
    df = pd.read_pickle(FEATURES_PKL)
    sig, v4n, pn = info["signal_names"], info["v4_names"], info["pooled_names"]
    sigfeat = {f"f_{s}_{st}" for s in sig for st in STATS}
    non_signal = [n for n in v4n if n not in sigfeat]
    cols = {"v4": v4n, "landmarks": pn + non_signal, "both": v4n + pn}[features]
    n_sig = len(sig)
    ch = {"v4": list(range(n_sig)), "landmarks": list(range(n_sig, n_sig + len(GROUPS))), "both": list(range(n_sig + len(GROUPS)))}[features]
    seq = np.load(SEQ_NPY, mmap_mode="r")
    ind = [f"f_{s}_mean" for s in sig] if features != "landmarks" else []
    ind += [f"p_{g}_mean_std" for g in GROUP_NAMES] if features != "v4" else []
    extra = [n for n in cols if n not in sigfeat]
    return dict(meta=df[["participant_id", "angle", "symptom", "video_id", "win_start_s"]].reset_index(drop=True),
                X=df[cols].to_numpy(np.float32), seq=np.asarray(seq[:, ch, :]), names=cols, features=features,
                ind_idx=[cols.index(n) for n in ind], extra_idx=[cols.index(n) for n in extra],
                track=df[[f"f_{fl}_rate" for fl in FLAGS]].to_numpy(np.float32), people=info["participants"],
                channel_names=[(sig + GROUP_NAMES)[i] for i in ch])


def class_weights(y, k, cap=10.0):
    n = np.bincount(y, minlength=k).astype(float)
    return np.clip(len(y) / (k * np.maximum(n, 1)), 0.1, cap).astype(np.float32)


def prf(y, pred, k):
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    p, r = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    return p, r, 2 * p * r / np.maximum(p + r, 1e-12)


class Tracker:
    def __init__(self, y, patience, verbose=False, tag=""):
        self.y, self.patience, self.verbose, self.tag = y, patience, verbose, tag
        self.best, self.epoch, self.state, self.wait = -1.0, 0, None, 0

    def step(self, epoch, prob, state_fn, extra=""):
        p, r, f = (float(a[1]) for a in prf(self.y, prob.argmax(1), 2))
        if self.verbose:
            print(f"{self.tag}epoch {epoch:3d} {extra} | val precision={p:.3f} recall={r:.3f} F1={f:.3f}", flush=True)
        if f > self.best:
            self.best, self.epoch, self.state, self.wait = f, epoch, state_fn(), 0
        else:
            self.wait += 1
        return self.wait >= self.patience


def binary_metrics(y, p, thr=THRESHOLD):
    from sklearn.metrics import roc_auc_score
    y, pred = np.asarray(y).astype(int), (np.asarray(p) >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    tn, fn = int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    pr, rc, sp = tp / max(tp + fp, 1), tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return dict(n=len(y), n_pos=int(y.sum()), tp=tp, fp=fp, tn=tn, fn=fn, precision=pr, recall=rc, specificity=sp,
                f1=2 * pr * rc / max(pr + rc, 1e-12), accuracy=(tp + tn) / max(len(y), 1), balanced_accuracy=(rc + sp) / 2,
                roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"))


def fold_map(people):
    ps = sorted(set(people))
    perm = np.random.default_rng(SEED).permutation(len(ps))
    return {ps[i]: r % N_FOLDS for r, i in enumerate(perm)}


def _inner_split(pid, tr_all, rng):
    people = np.array(sorted(set(pid[tr_all])))
    val = set(rng.choice(people, max(1, int(round(0.2 * len(people)))), replace=False).tolist()) if len(people) > 1 else set()
    isva = np.array([p in val for p in pid[tr_all]])
    return tr_all[~isva], tr_all[isva]


def cv_symptom(D, sym, fit_predict, tag=""):
    meta = D["meta"]
    cls, pid = meta["symptom"].to_numpy(), meta["participant_id"].to_numpy()
    fm = fold_map(D["people"])
    fold = np.array([fm[p] for p in pid])
    binary, yb = (cls == sym) | (cls == "control"), (cls == sym).astype(int)
    rng = np.random.default_rng(SEED)
    out = dict(idx=[], prob=[], oidx=[], oprob=[], infos=[])
    for f in range(N_FOLDS):
        te, oth = np.flatnonzero(binary & (fold == f)), np.flatnonzero(~binary & (fold == f))
        tr, va = _inner_split(pid, np.flatnonzero(binary & (fold != f)), rng)
        assert not (set(pid[te]) & set(pid[np.r_[tr, va]])), "participant leak"
        if len(np.unique(yb[np.r_[tr, va]])) < 2 or len(te) == 0:
            prior = np.bincount(yb[np.r_[tr, va]], minlength=2) / max(len(tr) + len(va), 1)
            (pt, po), info = [np.tile(prior, (len(s), 1)) for s in (te, oth)], dict(note="single-class fold: prior used")
        else:
            (pt, po), info = fit_predict(tr, va, [te, oth])
        m = binary_metrics(yb[te], pt[:, 1]) if len(te) else dict(precision=0, recall=0, f1=0, accuracy=0)
        print(f"{tag}fold {f}: windows={len(te)} | precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"F1={m['f1']:.3f} accuracy={m['accuracy']:.3f}", flush=True)
        out["idx"].append(te), out["prob"].append(pt[:, 1]), out["oidx"].append(oth), out["oprob"].append(po[:, 1])
        out["infos"].append(dict(fold=f, windows=len(te), precision=m["precision"], recall=m["recall"], f1=m["f1"],
                                 accuracy=m["accuracy"], **info))
    return {k: (np.concatenate(v) if k != "infos" else v) for k, v in out.items()}


def _clean(o):
    if isinstance(o, dict):
        return {str(a): _clean(b) for a, b in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    return o


def out_dir(tag):
    d = OUT / "results" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def symptom_report(D, sym, res, d):
    meta = D["meta"]
    cls, vid, pid, ang = (meta[c].to_numpy() for c in ("symptom", "video_id", "participant_id", "angle"))
    yb = (cls == sym).astype(int)
    idx, p = res["idx"], res["prob"]
    y, pred = yb[idx], (p >= THRESHOLD).astype(int)
    m = binary_metrics(y, p)
    (d / sym).mkdir(exist_ok=True)
    pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], index=["true_control", f"true_{sym}"],
                 columns=["pred_control", f"pred_{sym}"]).to_csv(d / sym / "confusion_matrix.csv")
    clip = pd.DataFrame({"v": vid[idx], "y": y, "p": p}).groupby("v").agg(y=("y", "first"), p=("p", "mean"))
    mc = binary_metrics(clip["y"], clip["p"])
    m["clip"] = {k: mc[k] for k in ("precision", "recall", "f1", "accuracy")}
    m["by_angle"] = {a: {k: v for k, v in binary_metrics(y[mm], p[mm]).items() if k in ("n", "precision", "recall", "f1", "accuracy")}
                     for a in np.unique(ang[idx]) for mm in [ang[idx] == a]}
    m["f1_fold_std"] = float(np.std([i["f1"] for i in res["infos"]]))
    m["folds"] = res["infos"]
    pv = float(y.mean())
    m["always_positive_f1"] = 2 * pv / (1 + pv)
    fr = pd.DataFrame(dict(video_id=vid[idx], participant_id=pid[idx], angle=ang[idx], task_class=np.where(y == 1, sym, "control"),
                           failed=(pred != y)))
    trk = pd.DataFrame(D["track"][idx], columns=["face_tracked", "pose_tracked", "left_hand_tracked", "right_hand_tracked"])
    fr = pd.concat([fr, trk], axis=1)
    keys = ["video_id", "participant_id", "angle", "task_class"]
    vids = fr.groupby(keys).agg(windows=("failed", "size"), failed=("failed", "sum"), fail_frac=("failed", "mean"),
                                face_tracked=("face_tracked", "mean"), pose_tracked=("pose_tracked", "mean"),
                                left_hand_tracked=("left_hand_tracked", "mean"), right_hand_tracked=("right_hand_tracked", "mean")).reset_index()
    vids["outlier"] = (vids["fail_frac"] >= OUTLIER_FRAC) & (vids["windows"] >= MIN_OUTLIER_WINDOWS)
    vids.insert(0, "symptom_model", sym)
    vids.sort_values("fail_frac", ascending=False).to_csv(d / sym / "outlier_videos.csv", index=False)
    people = fr.groupby(["participant_id", "task_class"]).agg(windows=("failed", "size"), fail_frac=("failed", "mean")).reset_index()
    people["outlier"] = (people["fail_frac"] >= OUTLIER_FRAC) & (people["windows"] >= MIN_OUTLIER_WINDOWS)
    people.insert(0, "symptom_model", sym)
    angles = fr.groupby(["angle", "task_class"]).agg(windows=("failed", "size"), fail_frac=("failed", "mean")).reset_index()
    angles.insert(0, "symptom_model", sym)
    rows = []
    for c in CLASSES:
        sel = (cls[idx] == c) if c in (sym, "control") else None
        pp = p[sel] if sel is not None else res["oprob"][cls[res["oidx"]] == c]
        rows.append(dict(model=sym, windows_of=c, n=len(pp), positive_rate=float((pp >= THRESHOLD).mean()) if len(pp) else float("nan")))
    (d / sym / "metrics.json").write_text(json.dumps(_clean(m), indent=1), encoding="utf-8")
    pd.DataFrame(dict(video_id=vid[idx], participant_id=pid[idx], angle=ang[idx], true=y, prob=p)).to_csv(d / sym / "oof_predictions.csv", index=False)
    print(f"[{sym}] precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f} accuracy={m['accuracy']:.3f} "
          f"AUC={m['roc_auc']:.3f} | TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}", flush=True)
    return dict(sym=sym, m=m, videos=vids, people=people, angles=angles, cross=pd.DataFrame(rows))


def confusion_figure(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: confusion_matrices.png skipped (CSV files were still written).")
        return
    n = len(results)
    cols = min(4, n)
    fig, axes = plt.subplots(-(-n // cols), cols, figsize=(3.6 * cols, 3.4 * (-(-n // cols))), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), results):
        m = r["m"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.axis("on")
        ax.imshow(pct, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", color="white" if pct[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["control", "symptom"])
        ax.set_yticks([0, 1], ["control", "symptom"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{r['sym']}\nF1 {m['f1']:.2f}  acc {m['accuracy']:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_summary(results, d):
    rows = []
    for r in results:
        m = r["m"]
        rows.append(dict(symptom=r["sym"], windows=m["n"], symptom_windows=m["n_pos"], precision=m["precision"], recall=m["recall"],
                         f1=m["f1"], accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"], roc_auc=m["roc_auc"],
                         tn=m["tn"], fp=m["fp"], fn=m["fn"], tp=m["tp"], clip_f1=m["clip"]["f1"], clip_accuracy=m["clip"]["accuracy"],
                         fold_f1_std=m["f1_fold_std"], always_positive_f1=m["always_positive_f1"]))
    t = pd.DataFrame(rows)
    t.to_csv(d / "summary_by_symptom.csv", index=False)
    pd.concat([r["videos"] for r in results]).sort_values(["symptom_model", "fail_frac"], ascending=[True, False]).to_csv(d / "all_videos_failure_rates.csv", index=False)
    flagged = pd.concat([r["videos"] for r in results])
    flagged[flagged["outlier"]].sort_values(["symptom_model", "fail_frac"], ascending=[True, False]).to_csv(d / "outlier_videos_all_symptoms.csv", index=False)
    people = pd.concat([r["people"] for r in results])
    people.to_csv(d / "failure_by_participant.csv", index=False)
    angles = pd.concat([r["angles"] for r in results])
    angles.to_csv(d / "failure_by_angle.csv", index=False)
    cross = pd.concat([r["cross"] for r in results])
    cross.to_csv(d / "cross_symptom_positive_rates.csv", index=False)
    confusion_figure(results, d / "confusion_matrices.png")

    print("\nOUT-OF-FOLD RESULTS (every participant is tested once, by a model that never saw them; threshold 0.5)")
    show = t[["symptom", "precision", "recall", "f1", "accuracy", "roc_auc", "tn", "fp", "fn", "tp"]].round(3)
    print(show.to_string(index=False))
    n80, n85, n90 = (int((t["f1"] >= x).sum()) for x in (0.80, 0.85, 0.90))
    print(f"MACRO precision={t['precision'].mean():.3f} recall={t['recall'].mean():.3f} F1={t['f1'].mean():.3f} accuracy={t['accuracy'].mean():.3f} | "
          f"F1 >=0.80: {n80}/{len(t)}, >=0.85: {n85}/{len(t)}, >=0.90: {n90}/{len(t)} | always-positive F1 about {t['always_positive_f1'].mean():.3f}, chance accuracy 0.5")

    print(f"\nOUTLIERS: videos where {OUTLIER_FRAC:.0%} or more of the windows were classified wrongly (at least {MIN_OUTLIER_WINDOWS} windows)")
    for r in results:
        v = r["videos"]
        f = v[v["outlier"]]
        pp = r["people"]
        pf = pp[pp["outlier"]]
        sym_n, ctl_n = int((f["task_class"] != "control").sum()), int((f["task_class"] == "control").sum())
        print(f"[{r['sym']}] flagged videos {len(f)}/{len(v)} (symptom videos {sym_n}, control videos {ctl_n}) | "
              f"flagged people: {', '.join(sorted(set(pf['participant_id'])))[:150] or 'none'}")
    both = people[people["outlier"] & (people["task_class"] != "control")].groupby("participant_id")["symptom_model"].nunique()
    many = both[both >= 3].sort_values(ascending=False)
    print("People whose symptom videos fail (>=50%) in 3 or more symptom models: " + (", ".join(f"{p} ({n}/7)" for p, n in many.items()) or "none"))
    pa = angles.pivot_table(index="symptom_model", columns="angle", values="fail_frac", aggfunc="mean").round(2)
    print("\nShare of windows classified wrongly, by camera angle:")
    print(pa.to_string())
    ct = cross.pivot(index="model", columns="windows_of", values="positive_rate").reindex(index=[r["sym"] for r in results], columns=CLASSES).round(2)
    print("\nHow often each model says 'present' on each kind of window (diagonal = recall; control column = false-alarm rate):")
    print(ct.to_string())
    print(f"\nFiles: {d}")
    return t
