"""MAS -> MediaPipe Pose -> per-frame landmark CSV extraction.

Usage:
    python -m symptom_fidgeting.extract_landmarks_fidgeting [--limit N] [--clips a.mp4,b.mp4]

Reads MAS annotations.json + jsons/videos.json, applies the prototype's
positive/negative/ignored label policy, runs MediaPipe Pose Landmarker
(VIDEO mode) over each local clip, and writes one CSV per clip under
symptom_fidgeting/data_extracted/.

CSV schema (one row per frame):
  dataset, clip, source_group, label, frame, timestamp_ms,
  pose_detected, lm_<joint>_{x,y}, lm_conf_<joint>, scale_ref
"""
import argparse
import csv
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from . import paths
from .dataset_map import usable_dataset, load_clip_to_source
from .pose_utils import (make_pose_landmarker, LM, EXTRACT_JOINTS,
                         DETECTED_COL, CONF_PREFIX, _col)

FIELDNAMES = (["dataset", "clip", "source_group", "label", "frame",
               "timestamp_ms", DETECTED_COL]
              + [_col(j, a) for j in EXTRACT_JOINTS for a in ("x", "y")]
              + [CONF_PREFIX + j for j in EXTRACT_JOINTS]
              + ["scale_ref"])


def _write_header(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(FIELDNAMES)


def extract_clip(video_path, clip_name, source_group, label, landmarker,
                 out_path):
    """Run pose detection over one clip; write its CSV. Returns stats dict."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "reason": "cannot-open"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []
    frame_idx = 0
    n_detected = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        ts_ms = int(round(frame_idx * 1000.0 / fps))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] > 640:
            rgb = cv2.resize(rgb, (640, 360))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        row = {"dataset": "MAS", "clip": clip_name,
               "source_group": source_group, "label": label,
               "frame": frame_idx, "timestamp_ms": ts_ms,
               DETECTED_COL: 0.0}
        for j in EXTRACT_JOINTS:
            row[_col(j, "x")] = row[_col(j, "y")] = ""
            row[CONF_PREFIX + j] = ""
        row["scale_ref"] = ""

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            row[DETECTED_COL] = 1.0
            n_detected += 1
            for j in EXTRACT_JOINTS:
                lm = lms[LM[j]]
                row[_col(j, "x")] = round(lm.x, 6)
                row[_col(j, "y")] = round(lm.y, 6)
                row[CONF_PREFIX + j] = round(lm.visibility, 5)
            row["scale_ref"] = round(
                abs(lms[LM["l_shoulder"]].x - lms[LM["r_shoulder"]].x), 6)
        rows.append(row)
        frame_idx += 1
    cap.release()

    if not rows:
        return {"status": "error", "reason": "no-frames"}
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    return {"status": "ok", "frames": frame_idx,
            "detected": n_detected,
            "det_rate": n_detected / frame_idx if frame_idx else 0.0}


def _run(clips, annotations, clip2src, out_dir, label_of, report_prefix=""):
    os.makedirs(out_dir, exist_ok=True)
    stats = {"attempted": 0, "ok": 0, "failed": [], "det_rates": []}
    t0 = time.time()
    for clip in clips:
        stats["attempted"] += 1
        video = os.path.join(paths.MAS_CLIPS_DIR, clip)
        label = label_of(clip)
        out = os.path.join(out_dir, clip.replace(".mp4", ".csv"))
        landmarker = make_pose_landmarker()
        r = extract_clip(video, clip, clip2src[clip], label, landmarker, out)
        del landmarker
        if r["status"] == "ok":
            stats["ok"] += 1
            stats["det_rates"].append(r["det_rate"])
            print(f"  [ok]   {clip}  frames={r['frames']:>4}  "
                  f"det={r['det_rate']*100:5.1f}%")
        else:
            stats["failed"].append((clip, r["reason"]))
            print(f"  [FAIL] {clip}  {r['reason']}")
    stats["seconds"] = time.time() - t0
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="only extract the first N usable clips (test mode)")
    ap.add_argument("--clips", default=None,
                    help="comma-separated exact clip filenames to extract")
    ap.add_argument("--out", default=paths.EXTRACTED_DIR)
    args = ap.parse_args(argv)

    pos, neg, ignored, annotations, clip2src = usable_dataset()
    print(f"Usable MAS subset: {len(pos)} positive / {len(neg)} negative "
          f"/ {len(ignored)} ignored")

    if args.clips:
        wanted = [c.strip() for c in args.clips.split(",")]
        sel = [(c, "pos" if c in pos else ("neg" if c in neg else None))
               for c in wanted]
        missing = [c for c, k in sel if k is None]
        if missing:
            print("Not usable (check spelling/label policy):", missing)
        clips = [c for c, k in sel if k is not None]
    else:
        # Balanced interleave so a --limit subset contains both classes.
        order = []
        for i in range(max(len(pos), len(neg))):
            if i < len(pos):
                order.append(pos[i])
            if i < len(neg):
                order.append(neg[i])
        clips = order
        if args.limit:
            clips = clips[:args.limit]

    label_of = lambda c: 1 if "body stress" in annotations[c] else 0
    stats = _run(clips, annotations, clip2src, args.out, label_of)

    print(f"\nExtraction done: attempted={stats['attempted']} "
          f"ok={stats['ok']} failed={len(stats['failed'])}")
    for c, reason in stats["failed"]:
        print(f"  FAILED {c}: {reason}")
    if stats["det_rates"]:
        rates = np.array(stats["det_rates"])
        print(f"detection rate: mean={rates.mean()*100:.1f}% "
              f"min={rates.min()*100:.1f}% max={rates.max()*100:.1f}%")
    print(f"total extraction time: {stats['seconds']:.1f}s "
          f"({stats['seconds']/max(stats['ok'],1):.2f}s/clip)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
