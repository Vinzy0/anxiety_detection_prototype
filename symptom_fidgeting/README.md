# Fidgeting Prototype (upper-body movement classifier)

A first working prototype that detects **fidgeting-like upper-body movement**
from video, built for the MAS dataset (thesis: *Real-Time Anxiety Symptom
Detection*).

> ⚠️ **What this is NOT:** this prototype does NOT diagnose anxiety, and it
> does NOT claim that the MAS `body stress` label is clinically identical to
> fidgeting. It detects movement patterns associated with the MAS dataset's
> body-stress/fidgeting **proxy** (see [Dataset & label policy](#dataset--label-policy)).

---

## Pipeline

```
MAS video clip
   -> MediaPipe Pose Landmarker (Lite, VIDEO mode)
   -> per-frame pose landmark CSV
   -> 30-frame sliding windows (step 5)
   -> temporal movement features (normalized by shoulder width)
   -> trained MLP classifier (focal loss)
   -> fidget probability  ->  FIDGET / NOT-FIDGET
```

The features describe **movement through time**, not static pose: wrist /
elbow / shoulder / hip motion relative to a stable body reference (e.g.
wrist-vs-shoulder), summarized per window by total displacement, mean/peak
velocity, velocity spread, direction changes, position spread, plus
finger-fist distance change and pose-quality flags. All distances are divided
by the person's shoulder width so the model does not learn camera distance /
body size / video resolution.

---

## Dataset & label policy

Dataset: **MAS** (official ScazLab clone, `MAS_dataset/MAS/clips`, 316 local
clips). Each clip maps (via `jsons/videos.json`) to its original source
YouTube video — this mapping is the **leakage group**.

Label policy for this prototype:

| Class | MAS annotation |
|---|---|
| **FIDGET (positive)** | contains `"body stress"` (even with other labels) |
| **NOT-FIDGET (negative)** | exactly `["no stress"]` |
| **IGNORED (not used)** | any other stress-only clip, e.g. `["audio stress"]` |

Facts about the usable local subset: **26 positive / 191 negative / 99
ignored**, spread over **~110 source videos**, of which **21 sources contain
BOTH classes**.

### Honest limitations

- `body stress` is a **proxy** for observable fidgeting; positives frequently
  co-occur with audio/face/prosodic stress, so the model may learn correlated
  signals, not pure movement.
- Class imbalance is severe (26 vs 191 clips). We use focal loss
  (γ=2, α=0.75) plus threshold tuning; reported positive-class precision /
  recall / F1 reflect that.

---

## Repository files (`symptom_fidgeting/`)

| File | Purpose |
|---|---|
| `paths.py` | Central paths + window/threshold config |
| `dataset_map.py` | MAS label policy + clip→source-video group mapping |
| `pose_utils.py` | MediaPipe Pose Landmarker setup + shared landmark dict |
| `features.py` | **Shared** temporal feature engineering (training + inference) |
| `extract_landmarks_fidgeting.py` | Video → pose CSV extraction |
| `validate_features.py` | Pre-training data/feature sanity checks |
| `train_fidgeting.py` | Group-safe training + eval + artifact saving |
| `inference_fidgeting.py` | Production inference from saved artifacts |
| `e2e_heldout_test.py` | Held-out end-to-end test (fresh process) |
| `models/` | Saved artifacts (see below) |
| `data_extracted/` | Per-clip pose CSVs (gitignored) |
| `experiments/` | Logs + run summaries (gitignored) |

`mediapipe_models/pose_landmarker_lite.task` is the official MediaPipe **Pose
Landmarker (Lite, float16)** — auto-downloaded on first use.

---

## Running extraction (MAS required)

```bash
# small smoke test first
.venv/Scripts/python -m symptom_fidgeting.extract_landmarks_fidgeting --limit 8

# full usable subset (26 pos + 191 neg)
.venv/Scripts/python -m symptom_fidgeting.extract_landmarks_fidgeting
```

Each clip becomes `symptom_fidgeting/data_extracted/<clip>.csv`.

## Feature validation (optional but recommended)

```bash
.venv/Scripts/python -m symptom_fidgeting.validate_features
```

## Training

```bash
.venv/Scripts/python -m symptom_fidgeting.train_fidgeting
```

Reads every CSV in `data_extracted/`, builds windows, splits **by source
video group** (`GroupShuffleSplit`, 20% of groups held out — group overlap
asserted `= 0`), trains, evaluates, and writes artifacts to
`symptom_fidgeting/models/`.

## Inference on a NEW video (no MAS, no retraining)

```bash
.venv/Scripts/python -m symptom_fidgeting.inference_fidgeting path/to/video.mp4
# optionally: --no-tflite  (use the .keras model instead)
```

## End-to-end held-out test

```bash
.venv/Scripts/python -m symptom_fidgeting.e2e_heldout_test
```

Re-runs saved-artifact inference over held-out validation clips and
cross-checks the TFLite interpreter against the Keras model.

---

## Using the pretrained model WITHOUT retraining

1. Pull the repo (artifacts in `symptom_fidgeting/models/` are committed).
2. Install deps: `pip install tensorflow mediapipe opencv-python numpy pandas scikit-learn`.
3. Run inference:
   ```python
   from symptom_fidgeting.inference_fidgeting import FidgetDetector
   det = FidgetDetector()                      # loads models/ from disk
   windows, prob, pred, dt, fps = det.predict_video("my_video.mp4")
   # prob ~= mean window fidget probability, pred = 1 if prob > threshold
   ```
   `FidgetDetector` loads the TFLite model by default. `models/` contains
   everything needed: `fidget_model.tflite` (+ `.keras`), `scaler_mean.npy`,
   `scaler_scale.npy`, `threshold.npy`, `window_frames.npy`,
   `window_step.npy`, `feature_names.json`, `metadata.json`.

---

## Leakage prevention

Clips from the same source YouTube video (same person/session) are NEVER
split across train and validation. The split is done by **source video
group**; the trainer prints and asserts `GROUP OVERLAP = 0` before training.

---

## Final validation metrics (actual run)

> (filled after training — see `symptom_fidgeting/models/metadata.json`)

| Metric | Value |
|---|---|
| Accuracy | — |
| Precision (positive) | — |
| Recall (positive) | — |
| F1 (positive) | — |
| ROC AUC | — |
| Confusion matrix | — |
| Train / val windows | — / — |
| Train / val source groups | — / — |
| Group overlap | 0 |

---

## Known limitations

- Small positive class (26 clips / ~10 source people); validation results
  should be read per-group, not as a single number.
- MAS clips are short (3–10 s); a 30-frame window at 24 fps ≈ 1.25 s of
  motion.
- Pose is only detected while the person faces the camera (torso visible);
  windows with < 60% detection are dropped.
- `body stress` is a proxy; multi-label contamination is inherent to MAS.
- CPU-only TensorFlow was used; the TFLite model is the deployment artifact.
