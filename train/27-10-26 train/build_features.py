import argparse
import time

import numpy as np
import pandas as pd

from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-videos", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(INPUT_CSV, nrows=0).columns.tolist()
    bs = [c for c in header if c.startswith("bs_")]
    fm = [c for c in header if c.startswith("face_matrix")]
    cols = raw_columns(bs, fm)
    missing = [c for c in cols if c not in header]
    if missing:
        raise ValueError(f"CSV lacks columns: {missing[:10]}")
    metas, feats, seqs, sig_names, people, dropped = [], [], [], None, set(), 0
    for n, v in enumerate(videos(INPUT_CSV, cols), 1):
        if a.limit_videos and n > a.limit_videos:
            break
        v = v.sort_values("frame").reset_index(drop=True)
        cls = str(v["symptom"].iloc[0]).strip().lower().replace(" ", "_")
        if cls not in CLASSES:
            raise ValueError(f"Unknown folder category {cls!r}")
        people.add(str(v["participant_id"].iloc[0]))
        ts = pd.to_numeric(v["timestamp_ms"], errors="coerce").to_numpy(np.float64)
        keep = np.isfinite(ts) & (ts > np.maximum.accumulate(np.r_[-np.inf, np.nan_to_num(ts[:-1], nan=-np.inf)]))
        dropped += int((~keep).sum())
        v, ts = v[keep].reset_index(drop=True), ts[keep]
        if len(v) < 2:
            continue
        raw = video_raw(v, bs, fm, (ts - ts[0]) / 1000.0)
        sig_names = raw["signal_names"]
        starts = default_starts(raw["t"])
        if len(starts) == 0:
            continue
        v4f, pf, G, E = window_all(raw, starts)
        for s in starts:
            metas.append((v["participant_id"].iloc[0], v["angle"].iloc[0], cls, v["video_id"].iloc[0], float(s)))
        feats.append(np.hstack([v4f, pf]).astype(np.float32))
        seqs.append(np.concatenate([G, E], axis=1).astype(np.float16))
        if n % 25 == 0:
            print(f"videos {n} windows {sum(len(x) for x in feats):,} ({time.time() - t0:.0f}s)", flush=True)
    names4 = v4_names(sig_names, bs, fm)
    df = pd.concat([pd.DataFrame(metas, columns=["participant_id", "angle", "symptom", "video_id", "win_start_s"]),
                    pd.DataFrame(np.vstack(feats), columns=names4 + POOLED_NAMES)], axis=1)
    df.to_pickle(FEATURES_PKL)
    np.save(SEQ_NPY, np.concatenate(seqs))
    INFO_JSON.write_text(json.dumps(dict(v4_names=names4, pooled_names=POOLED_NAMES, signal_names=sig_names, group_names=GROUP_NAMES,
                                         blendshapes=bs, face_matrix=fm, participants=sorted(people), window=WINDOW, bands=BANDS,
                                         dropped_frames=dropped)))
    print(df.groupby("symptom").agg(windows=("video_id", "size"), clips=("video_id", "nunique"),
                                    participants=("participant_id", "nunique")).to_string())
    ctrl = set(df.loc[df["symptom"] == "control", "participant_id"])
    print(f"windows={len(df):,} | v4 features={len(names4)} + all-landmark features={len(POOLED_NAMES)} | signals={len(sig_names)} "
          f"+ landmark-group energy channels={len(GROUP_NAMES)} | participants={len(people)} without control={sorted(people - ctrl)} | "
          f"dropped frames={dropped} | {time.time() - t0:.0f}s")
    print(f"features >50% NaN: {int((df[names4 + POOLED_NAMES].isna().mean() > 0.5).sum())}/{len(names4) + len(POOLED_NAMES)} | {FEATURES_PKL}")


if __name__ == "__main__":
    main()
