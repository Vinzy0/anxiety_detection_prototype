"""End-to-end held-out test (§19): fresh process, saved artifacts only.

Run AFTER training. In a NEW Python process (no in-memory training model):
  1. reads symptom_fidgeting/models/val_clips.json (validation clips from
     the group-safe split — never seen during training),
  2. picks one held-out positive and one held-out negative MAS clip,
  3. runs the production inference path (video -> MediaPipe Pose -> windows
     -> same features -> scaler -> saved model),
  4. reports probability/prediction vs ground truth,
  5. cross-checks the TFLite interpreter against the .keras model on the
     same feature input (tolerance 1e-2).

Usage:
    python -m symptom_fidgeting.e2e_heldout_test
"""
import json
import os
import sys
import time

import numpy as np

from . import paths
from .inference_fidgeting import FidgetDetector


def pick_val_clips(model_dir, k=1):
    """Return (positive_clips, negative_clips) from val_clips.json."""
    with open(os.path.join(model_dir, "val_clips.json")) as f:
        meta = json.load(f)
    pos = [(c, m) for c, m in meta.items() if m["label"] == 1]
    neg = [(c, m) for c, m in meta.items() if m["label"] == 0]
    pos.sort(), neg.sort()
    return pos[:k], neg[:k]


def tflite_vs_keras(det):
    """Agreement check: run the same random feature input through both."""
    import tensorflow as tf
    from .inference_fidgeting import focal_loss_for_reload
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, len(det.features))).astype(np.float32)
    # keras path
    model = tf.keras.models.load_model(
        os.path.join(det.dir, "fidget_model.keras"),
        custom_objects={"focal_loss": focal_loss_for_reload()})
    p_keras = float(model.predict(x, verbose=0)[0][0])
    # tflite path
    det.interp.set_tensor(det.in_["index"], x)
    det.interp.invoke()
    p_tflite = float(det.interp.get_tensor(det.out["index"])[0][0])
    return p_keras, p_tflite, abs(p_keras - p_tflite)


def main(argv=None):
    model_dir = paths.MODELS_DIR
    det = FidgetDetector(model_dir, use_tflite=True)
    print(f"Detector reloaded from disk: tflite={det.use_tflite} "
          f"window={det.window} step={det.step} threshold={det.threshold:.3f} "
          f"features={len(det.features)}")

    # TFLite vs keras agreement.
    p_k, p_t, diff = tflite_vs_keras(det)
    print(f"\nTFLite vs Keras agreement on one input: "
          f"keras={p_k:.4f} tflite={p_t:.4f} diff={diff:.6f}")
    assert diff < 1e-2, "TFLite and Keras disagree!"

    pos, neg = pick_val_clips(model_dir, k=2)
    print(f"\nHeld-out validation clips available: "
          f"{len(pos) + len(neg)} (pos={len(pos)}, neg={len(neg)})")
    results = []
    for clip, meta in pos + neg:
        video = os.path.join(paths.MAS_CLIPS_DIR, clip)
        if not os.path.exists(video):
            print(f"  [SKIP] missing video {clip}")
            continue
        t0 = time.time()
        windows, prob, pred, dt, fps = det.predict_video(video)
        results.append({"clip": clip, "group": meta["group"],
                        "true_label": meta["label"], "prob": prob,
                        "pred": pred,
                        "correct": int(pred == meta["label"]),
                        "windows": len(windows) if windows else 0,
                        "infer_seconds": round(dt, 2)})
        ok = "CORRECT" if pred == meta["label"] else "WRONG"
        print(f"\n  clip      : {clip}")
        print(f"  group     : {meta['group']}")
        print(f"  true      : {'FIDGET' if meta['label'] else 'NOT-FIDGET'}")
        print(f"  windows   : {results[-1]['windows']}")
        print(f"  prob      : {prob:.3f} (threshold {det.threshold:.2f})")
        print(f"  predicted : {'FIDGET' if pred else 'NOT-FIDGET'}  -> {ok}")
        print(f"  infer time: {dt:.2f}s")
    out_path = os.path.join(paths.EXPERIMENTS_DIR, "e2e_heldout_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results,
                   "threshold": det.threshold,
                   "tflite_vs_keras_diff": round(diff, 6)}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
