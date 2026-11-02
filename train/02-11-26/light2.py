import argparse

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from common import *

TRACK_LIMIT = 0.60
NORMAL_Z_LIMIT = 6.0
MIN_CALIBRATION_WINDOWS = 3
MAX_CALIBRATION_WINDOWS = 5
MIN_EVALUATION_CONTROL_WINDOWS = 3
SMOOTH_K = 3
THRESHOLDS = np.arange(0.25, 0.751, 0.01)
PRIMARY_VARIANT = "smooth3_nested_threshold"


def normalize_angle(value):
    return str(value).strip().casefold()


def personal_columns(names, sym):
    selected = []
    for i, raw in enumerate(names):
        n = raw.lower()
        if "detected" in n or n.endswith("_rate"):
            continue
        if sym in ("facial_tension", "fear"):
            use = n.startswith(("f_face_", "p_face_")) or "bs_" in n
        elif sym == "avoidant_gaze":
            use = any(k in n for k in ("face_head_", "face_gaze_", "face_nose_", "p_face_head", "p_face_all"))
        elif sym == "heavy_breathing":
            use = any(k in n for k in ("pose_shoulder", "pose_torso", "pose_11_", "pose_12_", "p_pose_torso", "p_pose_all"))
        elif sym == "knee_tremor":
            use = any(k in n for k in ("pose_25_", "pose_26_", "pose_27_", "pose_28_", "pose_29_", "pose_30_", "pose_31_", "pose_32_", "pose_leg", "p_pose_leg"))
        else:
            use = n.startswith(("f_left_hand_", "f_right_hand_", "p_lh_", "p_rh_")) or "left_hand_" in n or "right_hand_" in n
        if use:
            selected.append(i)
    if not selected:
        raise ValueError(f"No personal-baseline features found for {sym!r} in feature set {names[:5]} ...")
    return np.asarray(selected, dtype=int)


def robust_z(values, reference):
    med = np.nanmedian(reference, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(reference - med), axis=0)
    sd = np.nanstd(reference, axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 1e-4), mad, sd)
    scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0)
    return np.clip((values - med) / scale, -8.0, 8.0)


def choose_calibration_segments(meta, seed):
    """Reserve an initial Control segment per participant-angle, retaining later non-overlapping windows."""
    cls = meta["symptom"].astype(str).str.strip().str.lower().to_numpy()
    pid = meta["participant_id"].astype(str).to_numpy()
    vid = meta["video_id"].astype(str).to_numpy()
    angle = np.asarray([normalize_angle(x) for x in meta["angle"].astype(str)], dtype=object)
    starts = pd.to_numeric(meta["win_start_s"], errors="coerce").to_numpy(float)

    calibration = np.zeros(len(meta), dtype=bool)
    guard = np.zeros(len(meta), dtype=bool)
    assignments = []
    rng = np.random.default_rng(seed)
    win_seconds = float(WINDOW["win_s"])

    for person in sorted(np.unique(pid)):
        person_rows = np.flatnonzero(pid == person)
        for key in sorted(set(angle[person_rows])):
            angle_rows = person_rows[angle[person_rows] == key]
            control_rows = angle_rows[cls[angle_rows] == "control"]
            candidates = []

            for video in sorted(np.unique(vid[control_rows])):
                rows = control_rows[vid[control_rows] == video]
                rows = rows[np.argsort(starts[rows], kind="stable")]
                video_starts = starts[rows]
                if len(rows) < MIN_CALIBRATION_WINDOWS + 1 + MIN_EVALUATION_CONTROL_WINDOWS:
                    continue

                chosen_count = None
                chosen_guard = None
                chosen_end = None
                max_count = min(MAX_CALIBRATION_WINDOWS, len(rows))
                for count in range(max_count, MIN_CALIBRATION_WINDOWS - 1, -1):
                    calibration_end = video_starts[count - 1] + win_seconds
                    later = video_starts >= calibration_end - 1e-9
                    if int(later.sum()) >= MIN_EVALUATION_CONTROL_WINDOWS:
                        chosen_count = count
                        chosen_guard = rows[(np.arange(len(rows)) >= count) & ~later]
                        chosen_end = calibration_end
                        break

                if chosen_count is None:
                    continue
                cal_rows = rows[:chosen_count]
                eval_rows = rows[video_starts >= chosen_end - 1e-9]
                candidates.append((video, cal_rows, chosen_guard, len(eval_rows), chosen_end, len(rows)))

            if not candidates:
                assignments.append(dict(
                    participant_id=person,
                    calibration_angle=key.title(),
                    calibration_video_id="",
                    calibration_windows=0,
                    guard_windows=0,
                    control_videos=int(len(np.unique(vid[control_rows]))) if len(control_rows) else 0,
                    calibration_candidates=0,
                    evaluation_control_windows=int(len(control_rows)),
                    calibration_start_s=np.nan,
                    calibration_end_s=np.nan,
                    baseline_available=False,
                ))
                continue

            # Prefer the longest usable calibration recording; randomize only exact ties.
            longest = max(item[5] for item in candidates)
            finalists = [item for item in candidates if item[5] == longest]
            video, cal_rows, guard_rows, eval_in_video, cal_end, _ = finalists[int(rng.integers(0, len(finalists)))]
            calibration[cal_rows] = True
            guard[guard_rows] = True

            other_control_rows = control_rows[vid[control_rows] != video]
            evaluation_control_windows = int(len(other_control_rows) + eval_in_video)
            assignments.append(dict(
                participant_id=person,
                calibration_angle=key.title(),
                calibration_video_id=video,
                calibration_windows=int(len(cal_rows)),
                guard_windows=int(len(guard_rows)),
                control_videos=int(len(np.unique(vid[control_rows]))),
                calibration_candidates=int(len(candidates)),
                evaluation_control_windows=evaluation_control_windows,
                calibration_start_s=float(starts[cal_rows[0]]),
                calibration_end_s=float(cal_end),
                baseline_available=True,
            ))

    return calibration, guard, pd.DataFrame(assignments)


def quality_masks(D):
    t = np.nan_to_num(np.asarray(D["track"], dtype=np.float32), nan=0.0)
    face, pose, left, right = t.T
    hands = np.maximum(left, right)
    return t, {
        "facial_tension": face >= TRACK_LIMIT,
        "fear": face >= TRACK_LIMIT,
        "avoidant_gaze": face >= TRACK_LIMIT,
        "heavy_breathing": pose >= TRACK_LIMIT,
        "knee_tremor": pose >= TRACK_LIMIT,
        "hand_tremor": hands >= TRACK_LIMIT,
        "fidgeting": hands >= TRACK_LIMIT,
    }


def make_personal_matrix(D, sym, calibration_mask, excluded_mask):
    """Build personal z-scores using only the matching participant-angle Control segment."""
    X, meta = D["X"], D["meta"]
    ids = personal_columns(D["names"], sym)
    cls = meta["symptom"].astype(str).str.strip().str.lower().to_numpy()
    pid = meta["participant_id"].astype(str).to_numpy()
    angle = np.asarray([normalize_angle(x) for x in meta["angle"].astype(str).to_numpy()], dtype=object)
    z = np.full((len(X), len(ids)), np.nan, dtype=np.float32)
    available = np.zeros(len(X), dtype=np.float32)
    _, tracking_ok = quality_masks(D)
    good_reference = tracking_ok[sym]

    for person in np.unique(pid):
        person_rows = np.flatnonzero(pid == person)
        for angle_key in np.unique(angle[person_rows]):
            angle_rows = person_rows[angle[person_rows] == angle_key]
            ref_rows = angle_rows[calibration_mask[angle_rows] & (cls[angle_rows] == "control")]
            ref_rows = ref_rows[good_reference[ref_rows]]
            if len(ref_rows) < MIN_CALIBRATION_WINDOWS:
                continue

            target_rows = angle_rows[~excluded_mask[angle_rows]]
            if not len(target_rows):
                continue

            reference = X[ref_rows][:, ids]
            z[target_rows] = robust_z(X[target_rows][:, ids], reference)
            available[target_rows] = 1.0

    names = [f"personal_z_{D['names'][i]}" for i in ids] + ["personal_baseline_available"]
    return np.column_stack([z, available]).astype(np.float32), names, ids


def subset_dataset(D, rows):
    """Subset row-aligned arrays while preserving the original participant list."""
    out = dict(D)
    out["meta"] = D["meta"].iloc[rows].reset_index(drop=True)
    out["X"] = D["X"][rows]
    out["track"] = D["track"][rows]
    out.pop("seq", None)
    return out


def smooth_probabilities(meta, idx, probabilities, k=SMOOTH_K):
    if k <= 1 or len(idx) == 0:
        return np.asarray(probabilities, dtype=float).copy()
    frame = meta.iloc[idx][["participant_id", "video_id", "win_start_s"]].copy()
    frame["probability"] = np.asarray(probabilities, dtype=float)
    frame["position"] = np.arange(len(frame))
    frame = frame.sort_values(["participant_id", "video_id", "win_start_s"])
    frame["smoothed"] = frame.groupby(["participant_id", "video_id"], sort=False)["probability"].transform(
        lambda s: s.rolling(k, min_periods=1).mean()
    )
    output = np.empty(len(frame), dtype=float)
    output[frame["position"].to_numpy()] = frame["smoothed"].to_numpy()
    return output


def equal_prior_f1(y, probabilities, threshold):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(probabilities) >= float(threshold)
    tp = np.sum(pred & (y == 1))
    fn = np.sum((~pred) & (y == 1))
    fp = np.sum(pred & (y == 0))
    tn = np.sum((~pred) & (y == 0))
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return float(2.0 * tpr / max(1.0 + tpr + fpr, 1e-12))


def choose_threshold(y, probabilities):
    if len(np.unique(y)) < 2:
        return 0.5, float("nan")
    scores = np.asarray([equal_prior_f1(y, probabilities, t) for t in THRESHOLDS])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    chosen = best[np.argmin(np.abs(THRESHOLDS[best] - 0.5))]
    return float(THRESHOLDS[chosen]), float(scores[chosen])


def participant_video_weights(meta, rows, y):
    m = meta.iloc[rows]
    g = pd.DataFrame({
        "person": m["participant_id"].astype(str).to_numpy(),
        "video": m["video_id"].astype(str).to_numpy(),
        "label": np.asarray(y, dtype=int),
    })
    video_windows = g.groupby(["person", "label", "video"])["video"].transform("size").to_numpy()
    person_videos = g.groupby(["person", "label"])["video"].transform("nunique").to_numpy()
    w = 1.0 / (video_windows * person_videos)
    return (w / w.mean()).astype(np.float32)


def make_fit_predict(D, X, sym, weighted, filter_tracking):
    cls = D["meta"]["symptom"].astype(str).str.strip().str.lower().to_numpy()
    yb = (cls == sym).astype(int)
    _, masks = quality_masks(D)
    quality = masks[sym]

    def fit_predict(tr, va, sets):
        inner_train = np.asarray(tr, dtype=int)
        inner_valid = np.asarray(va, dtype=int)
        fit_inner = inner_train[quality[inner_train]] if filter_tracking else inner_train
        if len(np.unique(yb[fit_inner])) < 2:
            fit_inner = inner_train

        val_raw_f1 = val_smooth_f1 = float("nan")
        thr_raw = thr_smooth = 0.5
        if len(np.unique(yb[fit_inner])) == 2:
            val_params = dict(LGBM_PARAMS)
            val_weight = None
            if weighted:
                val_params["class_weight"] = None
                val_weight = participant_video_weights(D["meta"], fit_inner, yb[fit_inner])
            val_model = LGBMClassifier(**val_params)
            val_model.fit(X[fit_inner], yb[fit_inner], sample_weight=val_weight)
            p_valid = val_model.predict_proba(X[inner_valid])[:, 1]
            p_valid_smooth = smooth_probabilities(D["meta"], inner_valid, p_valid, SMOOTH_K)
            thr_raw, val_raw_f1 = choose_threshold(yb[inner_valid], p_valid)
            thr_smooth, val_smooth_f1 = choose_threshold(yb[inner_valid], p_valid_smooth)

        outer_train = np.r_[inner_train, inner_valid]
        fit_final = outer_train[quality[outer_train]] if filter_tracking else outer_train
        if len(np.unique(yb[fit_final])) < 2:
            fit_final = outer_train
        final_params = dict(LGBM_PARAMS)
        final_weight = None
        if weighted:
            final_params["class_weight"] = None
            final_weight = participant_video_weights(D["meta"], fit_final, yb[fit_final])
        model = LGBMClassifier(**final_params)
        model.fit(X[fit_final], yb[fit_final], sample_weight=final_weight)
        probabilities = [model.predict_proba(X[s]) if len(s) else np.zeros((0, 2)) for s in sets]
        info = dict(
            threshold_raw=thr_raw,
            threshold_smooth3=thr_smooth,
            inner_val_f1_50_50_raw=val_raw_f1,
            inner_val_f1_50_50_smooth3=val_smooth_f1,
            tracking_filtered=int(len(outer_train) - len(fit_final)),
        )
        return probabilities, info

    return fit_predict, yb, quality


def metrics_at_threshold(y, probabilities, threshold):
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    pred = probabilities >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum((~pred) & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    balanced_accuracy = (recall + specificity) / 2
    fpr = fp / max(fp + tn, 1)
    f1_50 = 2 * recall / max(1 + recall + fpr, 1e-12)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else float("nan")
    except Exception:
        auc = float("nan")
    return dict(
        n=len(y), n_pos=int(y.sum()), tp=tp, fp=fp, tn=tn, fn=fn,
        precision=precision, recall=recall, specificity=specificity,
        f1=f1, accuracy=(tp + tn) / max(len(y), 1),
        balanced_accuracy=balanced_accuracy, f1_50_50=f1_50,
        accuracy_50_50=balanced_accuracy, roc_auc=auc,
    )


def fold_thresholds(D, res, threshold_key):
    fm = fold_map(D["people"])
    idx = res["idx"]
    pid = D["meta"].iloc[idx]["participant_id"].astype(str).to_numpy()
    threshold_by_fold = {int(i["fold"]): float(i.get(threshold_key, 0.5)) for i in res["infos"]}
    fold_ids = np.asarray([fm[p] for p in pid], dtype=int)
    return np.asarray([threshold_by_fold[f] for f in fold_ids], dtype=float)


def evaluate_variants(D, sym, res):
    idx, p = res["idx"], res["prob"]
    cls = D["meta"]["symptom"].astype(str).str.strip().str.lower().to_numpy()
    y = (cls[idx] == sym).astype(int)
    p3 = smooth_probabilities(D["meta"], idx, p, SMOOTH_K)
    thr_raw = fold_thresholds(D, res, "threshold_raw")
    thr_smooth = fold_thresholds(D, res, "threshold_smooth3")
    specs = [
        ("raw_fixed_050", p, np.full(len(p), 0.5)),
        ("smooth3_fixed_050", p3, np.full(len(p3), 0.5)),
        ("raw_nested_threshold", p, thr_raw),
        (PRIMARY_VARIANT, p3, thr_smooth),
    ]
    rows = []
    for name, prob, threshold in specs:
        m = metrics_at_threshold(y, prob, threshold)
        rows.append(dict(
            symptom=sym, variant=name, **m,
            threshold_median=float(np.median(threshold)),
            threshold_min=float(np.min(threshold)),
            threshold_max=float(np.max(threshold)),
        ))
    return rows


def personal_outlier_report(D, sym, Z, personal_names):
    meta = D["meta"]
    abs_z = np.abs(Z[:, :-1])
    safe = np.where(np.isfinite(abs_z), abs_z, -1.0)
    max_z = safe.max(axis=1)
    max_z[max_z < 0] = np.nan
    best = safe.argmax(axis=1)
    top = np.asarray(personal_names[:-1], dtype=object)[best]
    top[~np.isfinite(max_z)] = ""
    out = meta[["video_id", "participant_id", "angle", "symptom", "win_start_s"]].copy()
    out.insert(0, "symptom_model", sym)
    out["baseline_available"] = Z[:, -1] > 0
    out["max_abs_control_z"] = max_z
    out["highest_deviation_feature"] = top
    out["beyond_normal_range"] = max_z >= NORMAL_Z_LIMIT
    return out


def write_quality_reports(D, root, personal_reports, assignments):
    meta = D["meta"].reset_index(drop=True)
    t, masks = quality_masks(D)
    q = meta[["video_id", "participant_id", "angle", "symptom", "win_start_s"]].copy()
    q["face_tracking"], q["pose_tracking"] = t[:, 0], t[:, 1]
    q["left_hand_tracking"], q["right_hand_tracking"] = t[:, 2], t[:, 3]
    for sym, mask in masks.items():
        q[f"tracking_usable_{sym}"] = mask
    q.to_csv(root / "tracking_quality_windows.csv", index=False)

    assignments.to_csv(root / "personal_calibration_assignments.csv", index=False)
    if personal_reports:
        windows = pd.concat(personal_reports, ignore_index=True)
        windows.to_csv(root / "normal_range_candidates.csv", index=False)
        windows["segment_start_s"] = (np.floor(windows["win_start_s"] / 10.0) * 10).astype(int)
        keys = ["symptom_model", "video_id", "participant_id", "angle", "symptom", "segment_start_s"]
        segments = windows.groupby(keys, dropna=False).agg(
            windows=("max_abs_control_z", "size"),
            median_deviation_z=("max_abs_control_z", "median"),
            max_deviation_z=("max_abs_control_z", "max"),
            beyond_normal_share=("beyond_normal_range", "mean"),
        ).reset_index()
        segments["review_segment"] = (segments["beyond_normal_share"] >= 0.5) & (segments["windows"] >= 2)
        segments.to_csv(root / "normal_range_segments_10s.csv", index=False)
        coverage = windows.groupby(["participant_id", "symptom_model", "angle"], dropna=False).agg(
            windows=("baseline_available", "size"),
            videos=("video_id", "nunique"),
            baseline_available=("baseline_available", "mean"),
        ).reset_index()
        coverage.to_csv(root / "personal_baseline_coverage.csv", index=False)


def append_calibration_info(arm_dir, sym, res, X, assignments):
    path = arm_dir / sym / "oof_predictions.csv"
    if not path.is_file():
        return
    pred = pd.read_csv(path)
    idx = res["idx"]
    pred["personal_baseline_available"] = (X[idx, -1] > 0).astype(int)

    lookup = {
        (str(row.participant_id), normalize_angle(row.calibration_angle)): str(row.calibration_video_id)
        for row in assignments.itertuples()
        if bool(row.baseline_available)
    }
    pred["calibration_video_id"] = [
        lookup.get((str(person), normalize_angle(angle)), "")
        for person, angle in zip(pred["participant_id"], pred["angle"])
    ]
    pred.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=FEATURE_SETS, default="both")
    ap.add_argument("--only", nargs="*", choices=SYMPTOMS)
    ap.add_argument("--arms", nargs="+", choices=["base", "personal", "personal_weighted"],
                    default=["base", "personal", "personal_weighted"])
    ap.add_argument("--calibration-seed", type=int, default=SEED)
    ap.add_argument("--filter-poor-tracking", action="store_true",
                    help="Filter low-tracking training windows only; test windows are never removed.")
    ap.add_argument("--no-final", action="store_true")
    a = ap.parse_args()

    D_full = load(a.features)
    meta_full = D_full["meta"].reset_index(drop=True)
    cls_full = meta_full["symptom"].astype(str).str.strip().str.lower().to_numpy()

    tag = f"vs_control_lgbm_angle_segment_nested_v1_{a.features}_seed{a.calibration_seed}"
    root = out_dir(tag)
    root.mkdir(parents=True, exist_ok=True)
    syms = a.only or SYMPTOMS

    calibration_mask, guard_mask, assignments = choose_calibration_segments(meta_full, a.calibration_seed)
    excluded_mask = calibration_mask | guard_mask
    eligible_rows = np.flatnonzero(~excluded_mask)
    if not len(eligible_rows):
        raise RuntimeError("No model rows remain after reserving calibration segments and overlap guards.")

    D = subset_dataset(D_full, eligible_rows)
    cls = cls_full[eligible_rows]
    X_base = D_full["X"][eligible_rows]
    _, quality = quality_masks(D)

    eval_control_count = int((cls == "control").sum())
    if eval_control_count == 0:
        raise RuntimeError("No Control windows remain for evaluation. Calibration reservation is invalid.")
    fold_assignment = fold_map(D["people"])
    participant_ids = D["meta"]["participant_id"].astype(str).to_numpy()
    row_folds = np.asarray([fold_assignment[p] for p in participant_ids], dtype=int)
    excluded_ids = np.isin(participant_ids, list(EXCLUDED_TRAINING_PARTICIPANTS))
    for sym in syms:
        positive_count = int((cls == sym).sum())
        if positive_count == 0:
            raise RuntimeError(f"No evaluation windows remain for symptom {sym!r}.")
        binary_rows = (cls == sym) | (cls == "control")
        for fold_id in range(N_FOLDS):
            test_rows = binary_rows & (row_folds == fold_id)
            train_rows = binary_rows & (row_folds != fold_id) & ~excluded_ids
            if len(np.unique((cls[test_rows] == sym).astype(int))) < 2:
                raise RuntimeError(
                    f"{sym}: outer fold {fold_id} has only one test class after calibration reservation."
                )
            if len(np.unique((cls[train_rows] == sym).astype(int))) < 2:
                raise RuntimeError(
                    f"{sym}: outer fold {fold_id} has only one training class. Check participant coverage and exclusions."
                )
        print(f"Evaluation class counts [{sym}]: control={eval_control_count:,} | symptom={positive_count:,}")

    reserved_pairs = int(assignments["baseline_available"].sum())
    reserved_videos = assignments.loc[assignments["baseline_available"], "calibration_video_id"].nunique()
    angle_counts = assignments.loc[assignments["baseline_available"]].groupby("calibration_angle")["participant_id"].nunique().to_dict()
    print(f"LightGBM | features={a.features} ({D_full['X'].shape[1]} columns) | windows={len(D_full['X']):,} | people={len(D_full['people'])}")
    print(f"Calibration: initial Control segment per participant-angle; later non-overlapping windows retained | seed={a.calibration_seed}")
    print(f"Reserved calibration windows={int(calibration_mask.sum()):,} | overlap-guard windows={int(guard_mask.sum()):,} | model/evaluation windows={len(eligible_rows):,}")
    print(f"Reserved participant-angle baselines={reserved_pairs} | distinct calibration videos={reserved_videos}")
    print("Calibration participants by angle: " + (", ".join(f"{k}={v}" for k, v in sorted(angle_counts.items())) or "none"))

    angle_coverage = assignments.groupby("participant_id")["baseline_available"].sum().value_counts().sort_index().to_dict()
    coverage_text = ", ".join(f"{count} angles={people} participants" for count, people in sorted(angle_coverage.items()))
    print(f"Participant angle coverage: {coverage_text or 'none'}")
    print(f"Evaluation Control windows retained: {int((cls == 'control').sum()):,}")
    unavailable = assignments.loc[~assignments["baseline_available"]]
    print(f"Participant-angles without usable Control calibration: {len(unavailable)}")
    excluded = sorted(str(x) for x in EXCLUDED_TRAINING_PARTICIPANTS)
    print(f"Excluded training participants: {excluded or 'none'}")
    print(f"arms={a.arms} | tracking filter={'ON' if a.filter_poor_tracking else 'OFF'} | causal smoothing={SMOOTH_K} windows")
    print("Tracking-usable windows: " + ", ".join(f"{s}={quality[s].mean():.1%}" for s in syms))

    personal = {}
    outlier_reports = []
    if any(x != "base" for x in a.arms):
        for sym in syms:
            Z_full, z_names, ids = make_personal_matrix(D_full, sym, calibration_mask, excluded_mask)
            Z = Z_full[eligible_rows]
            personal[sym] = (np.hstack([X_base, Z]).astype(np.float32), z_names, ids)
            outlier_reports.append(personal_outlier_report(D, sym, Z, z_names))
            available = Z[:, -1] > 0
            n_pairs = D["meta"].loc[available, ["participant_id", "angle"]].drop_duplicates().shape[0]
            print(f"[{sym}] personal features={len(ids)} | calibrated windows={int(available.sum()):,}/{len(Z):,} | participant-angle pairs={n_pairs}")

    write_quality_reports(D, root, outlier_reports, assignments)

    all_variant_rows, all_threshold_rows = [], []
    for arm in a.arms:
        arm_dir = root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        standard_results = []
        print(f"\n=== {arm} ===")
        for sym in syms:
            if arm == "base":
                X = X_base
                z_names, personal_ids = [], []
                weighted = False
            else:
                X, z_names, personal_ids = personal[sym]
                weighted = arm == "personal_weighted"

            fit_predict, yb, quality_mask = make_fit_predict(D, X, sym, weighted, a.filter_poor_tracking)
            res = cv_symptom(D, sym, fit_predict, f"[{sym}|{arm}] ")
            report = symptom_report(D, sym, res, arm_dir)
            standard_results.append(report)
            if arm != "base":
                append_calibration_info(arm_dir, sym, res, X, assignments)

            variant_rows = evaluate_variants(D, sym, res)
            all_variant_rows.extend([dict(arm=arm, **r) for r in variant_rows])
            for info in res["infos"]:
                all_threshold_rows.append(dict(arm=arm, symptom=sym, **{
                    k: info.get(k, np.nan) for k in ("fold", "threshold_raw", "threshold_smooth3",
                                                     "inner_val_f1_50_50_raw", "inner_val_f1_50_50_smooth3")
                }))

            if not a.no_final:
                pid = D["meta"]["participant_id"].astype(str).to_numpy()
                keep = ((cls == sym) | (cls == "control")) & ~np.isin(
                    pid, list(EXCLUDED_TRAINING_PARTICIPANTS)
                )
                rows = np.flatnonzero(keep)
                if a.filter_poor_tracking:
                    candidate = rows[quality_mask[rows]]
                    if len(candidate) and len(np.unique(yb[candidate])) == 2:
                        rows = candidate
                params = dict(LGBM_PARAMS)
                sample_weight = None
                if weighted:
                    params["class_weight"] = None
                    sample_weight = participant_video_weights(D["meta"], rows, yb[rows])
                final_model = LGBMClassifier(**params).fit(X[rows], yb[rows], sample_weight=sample_weight)
                med_threshold = float(np.median([i.get("threshold_smooth3", 0.5) for i in res["infos"]]))
                joblib.dump(dict(
                    model=final_model,
                    features=list(D_full["names"]) + z_names,
                    symptom=sym,
                    feature_set=a.features,
                    window=WINDOW,
                    threshold=med_threshold,
                    smoothing_windows=SMOOTH_K,
                    threshold_selection="Nested participant-grouped inner validation; equal-prior positive-class F1; threshold range 0.25-0.75.",
                    uses_personal_control_baseline=(arm != "base"),
                    personal_feature_indices=personal_ids,
                    personal_feature_names=z_names,
                    calibration_protocol="An initial Control segment of up to five windows is reserved per participant-angle. Any following windows overlapping the final calibration window are excluded; later non-overlapping windows remain available for training/evaluation. Each evaluation window uses only the matching participant-angle baseline.",
                    calibration_seed=a.calibration_seed,
                    min_calibration_windows=MIN_CALIBRATION_WINDOWS,
                    personal_method="Matching-angle, per-participant initial Control-segment median/MAD; robust z-score clipped to [-8, 8].",
                    tracking_filter=bool(a.filter_poor_tracking),
                    weighted_by_participant_video=weighted,
                ), arm_dir / sym / "model.joblib")
                print(f"[{sym}] final model trained on {len(rows):,} non-calibration windows | deployment threshold={med_threshold:.3f}")

        write_summary(standard_results, arm_dir)

    variants = pd.DataFrame(all_variant_rows)
    thresholds = pd.DataFrame(all_threshold_rows)
    variants.to_csv(root / "evaluation_variants_by_symptom.csv", index=False)
    thresholds.to_csv(root / "nested_fold_thresholds.csv", index=False)
    macro = variants.groupby(["arm", "variant"], as_index=False).agg(
        symptoms=("symptom", "nunique"),
        macro_precision=("precision", "mean"),
        macro_recall=("recall", "mean"),
        macro_f1=("f1", "mean"),
        macro_accuracy=("accuracy", "mean"),
        macro_f1_50_50=("f1_50_50", "mean"),
        macro_accuracy_50_50=("accuracy_50_50", "mean"),
        macro_roc_auc=("roc_auc", "mean"),
    )
    macro.to_csv(root / "evaluation_variants_macro.csv", index=False)

    print("\nEVALUATION COMPARISON")
    print(macro.round(3).to_string(index=False))
    primary = macro[macro["variant"] == PRIMARY_VARIANT].sort_values("macro_f1_50_50", ascending=False)
    if len(primary):
        print("\nPRIMARY PRE-DECLARED VARIANT: " + PRIMARY_VARIANT)
        print(primary[["arm", "macro_f1", "macro_f1_50_50", "macro_accuracy", "macro_accuracy_50_50"]].round(3).to_string(index=False))
    print("\nNatural-prevalence F1 uses the actual evaluation class proportions.")
    print("F1_50_50 standardizes each model's sensitivity and false-positive rate to equal class prevalence.")
    print("Accuracy_50_50 is balanced accuracy. All thresholds were selected from inner-validation participants only.")
    print("Initial calibration segments and overlapping guard windows are excluded from classifier training and evaluation; later non-overlapping Control windows are retained.")
    print(f"\nReports and models: {root}")


if __name__ == "__main__":
    main()
