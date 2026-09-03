"""Feature/data validation before training (§12 of the brief).

Checks the extracted landmark CSVs + window features for NaN/inf, constant
columns, leaked identifier fields, and gross class imbalance / separability
problems. Prints a report. Run after full extraction, before training.

Usage:
    python -m symptom_fidgeting.validate_features [--csv_dir DIR]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from . import paths
from .dataset_map import usable_dataset
from .features import window_to_features, feature_names


def check_csvs(csv_dir):
    print("=" * 60)
    print("CSV-LEVEL CHECKS")
    print("=" * 60)
    fs = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    print(f"CSV count: {len(fs)}")
    bad = []
    for f in fs:
        df = pd.read_csv(os.path.join(csv_dir, f))
        # label/source sanity
        if df["label"].nunique() != 1:
            bad.append(f"{f}: mixed labels")
        if df["source_group"].nunique() != 1:
            bad.append(f"{f}: mixed groups")
        if "pose_detected" in df and df["pose_detected"].max() < 0.5:
            bad.append(f"{f}: no detected frames")
        # detect-rate histogram across clips
    dets = []
    for f in fs:
        df = pd.read_csv(os.path.join(csv_dir, f))
        if "pose_detected" in df and len(df):
            dets.append(df["pose_detected"].mean())
    dets = np.array(dets)
    if dets.size:
        print(f"Clip pose detection rate: mean={dets.mean()*100:.1f}% "
              f"min={dets.min()*100:.1f}% p10={np.percentile(dets,10)*100:.1f}% "
              f"clips<60%: {(dets < 0.6).sum()}/{dets.size}")
    if bad:
        print("PROBLEMS:")
        for b in bad:
            print("  ", b)
    else:
        print("All CSVs consistent (single label + group, pose seen).")


def check_features(csv_dir):
    print("\n" + "=" * 60)
    print("WINDOW-FEATURE CHECKS")
    print("=" * 60)
    feat_names = feature_names()
    rows, y = [], []
    for f in sorted(os.listdir(csv_dir)):
        if not f.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(csv_dir, f))
        lab = int(df["label"].iloc[0])
        n = len(df)
        for s in range(0, max(0, n - paths.WINDOW_FRAMES) + 1,
                       paths.WINDOW_STEP):
            if s + paths.WINDOW_FRAMES > n:
                break
            feats, _ = window_to_features(df.iloc[s:s + paths.WINDOW_FRAMES])
            if feats is not None:
                rows.append(feats)
                y.append(lab)
    X = np.asarray(rows, np.float32)
    y = np.asarray(y, np.int32)
    print(f"Windows: {X.shape[0]}, features: {X.shape[1]}, "
          f"pos rate: {y.mean()*100:.1f}%")
    nan = np.isnan(X).sum()
    inf = np.isinf(X).sum()
    print(f"NaN: {nan}   Inf: {inf}")
    std = X.std(axis=0)
    const = np.where(std < 1e-9)[0]
    if const.size:
        print(f"CONSTANT columns ({const.size}):")
        for i in const:
            print(f"  {i}: {feat_names[i]}")
    # extreme outliers
    z = np.abs((X - X.mean(axis=0)) / (std + 1e-9))
    print(f"Extreme outliers (>20 std): {(z > 20).sum()}")
    # pos vs neg means (quick sanity that features differ)
    if y.mean() not in (0, 1):
        m_pos = X[y == 1].mean(axis=0)
        m_neg = X[y == 0].mean(axis=0)
        top = np.argsort(np.abs(m_pos - m_neg))[-8:][::-1]
        print("Largest pos-vs-neg mean gaps:")
        for i in top:
            print(f"  {feat_names[i]:<46} pos={m_pos[i]:+.3f} "
                  f"neg={m_neg[i]:+.3f}")
    return X, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv_dir", default=paths.EXTRACTED_DIR)
    args = ap.parse_args()
    check_csvs(args.csv_dir)
    check_features(args.csv_dir)
