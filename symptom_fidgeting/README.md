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
by the person's **shoulder width** so the model does not learn camera
distance / body size / video resolution.

### Normalization (shoulder width)

Every 30-frame window is scaled by the median **euclidean shoulder width**
over that window's valid frames:

```
width(frame) = hypot(l_shoulder_x - r_shoulder_x, l_shoulder_y - r_shoulder_y)
scale(window) = median(width(frame)) over pose-detected, finite,
                confidence>=0.3 shoulder frames
```

using both shoulders and both x and y (shoulders are rarely exactly
horizontal). Missing / NaN / low-confidence frames never move the scale.
Windows with no valid shoulder pair (no usable body scale) are **dropped** —
the model never silently scales by 1.0. v1 had a bug that always scaled by
1.0 (see History); the committed model is the corrected one.

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
| `tests/test_scale.py` | Regression test: shoulder-width normalization |
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

## Final validation metrics (corrected model — actual run)

Group-safe split: 20% of source groups held out. Windows of 30 frames /
step 5. Validation = 2,198 windows from 18 source groups (52 clips) never
seen during training. **Split caveat: only 3 of the 18 held-out groups are
positive** (David, Julian, Derrick), so headline numbers are not a
population-level estimate.

### Window-level metrics

| Metric | Value |
|---|---|
| Accuracy | 0.947 |
| Precision (positive) | 0.642 |
| Recall (positive) | 0.610 |
| F1 (positive) | 0.626 |
| ROC AUC | 0.938 |
| Confusion matrix (TN/FP/FN/TP) | 1,985 / 54 / 62 / 97 |
| Decision threshold | 0.62 |
| Train / val windows | 6,103 / 2,198 |
| Train / val source groups | 71 / 18 |
| Group overlap | **0** |
| Val positive rate | 7.2% |

### Clip-level metrics

Windows within a clip are highly correlated (2,198 windows ≠ 2,198
independent samples). Aggregating each clip's window probabilities to ONE
clip probability (mean) and comparing to the same threshold:

| Metric | Value |
|---|---|
| Validation clips | 52 (3 positive / 49 negative) |
| Accuracy | 0.962 |
| Precision (positive) | 0.667 |
| Recall (positive) | 0.667 |
| F1 (positive) | 0.667 |
| ROC AUC | 0.973 |
| Confusion matrix (TN/FP/FN/TP) | 48 / 1 / 1 / 2 |

Per-source-group numbers in `models/metadata.json` (window-level plus
clip-level) and `experiments/train_corrected.log`. Held-out group behavior
is uneven: David F1 0.92, Julian F1 0.66; Derrick is poorly recalled
(precision 0.80 / recall 0.08) — visible in both window and clip metrics.

### Generalization check (group-aware 5-fold CV)

Re-ran the same pipeline with `GroupKFold(5)` over source videos (details +
per-fold rows in `experiments/cv_corrected.json`) to gauge how unstable
performance is across different held-out groups — this is NOT a tuned
model, just stability evidence:

| Metric | window-level (mean ± std) | clip-level (mean ± std) |
|---|---|---|
| F1 (positive) | 0.520 ± 0.165 | 0.503 ± 0.067 |
| Recall (positive) | 0.731 ± 0.192 | 0.736 ± 0.182 |
| Precision (positive) | 0.412 ± 0.150 | 0.390 ± 0.050 |
| ROC AUC | 0.815 ± 0.031 | 0.832 ± 0.017 |

Averages are over the 4/5 folds that contained positive held-out groups.
Fold 1 happened to hold out zero positive groups — with only 26 positive
clips (~10 people), whether a fold has any positives at all is luck, and
fold-level F1 swings 0.24 → 0.67 with group composition. Treat any single
split's headline number with caution.

---

## Known limitations

- **Only 26 positive clips exist locally** (26 vs 191), from only ~10
  source people. Validation in the committed split covers just **3 positive
  held-out groups** (David, Julian, Derrick); per-window F1/AUC here are NOT
  a population-level estimate.
- **Window samples are correlated within clips** (2,198 windows from 52
  clips). Clip-level metrics (mean-window probability per clip) are the more
  honest headline; read both.
- **Threshold is tuned on the validation split** used for the headline
  metrics — results are mildly optimistic. Clip-level n is tiny (3
  positives).
- `body stress` is a **proxy**; multi-label contamination is inherent to
  MAS. Positives frequently co-occur with audio/face/prosodic stress, so the
  model may learn correlated signals, not pure movement.
- MAS clips are short (3–10 s); a 30-frame window at 24 fps ≈ 1.25 s of
  motion.
- Pose is only detected while the person faces the camera (torso visible);
  windows with < 60% detection are dropped.
- CPU-only TensorFlow was used; the TFLite model is the deployment artifact.

## History

**v1 (buggy, replaced):** the shoulder-width normalizer computed only the
right shoulder's x for both sides (`r_shoulder_x - r_shoulder_x = 0`) and
fell back to `1.0` on every window — the v1 model was NOT actually
shoulder-width normalized, and its metrics (val F1 0.685 / AUC 0.964) were
reported under an incorrect description. Replaced by the corrected model
below (val F1 0.626 / AUC 0.938 window-level).

## Regression test

`symptom_fidgeting/tests/test_scale.py` guards the normalizer:

```bash
.venv/Scripts/python -m unittest symptom_fidgeting.tests.test_scale -v
```

Synthetic windows verify known shoulder widths (including y-axis
contribution), median robustness to missing/low-confidence frames, and that
degenerate windows return `0` (dropped) rather than a spurious `1.0`. Real
extracted MAS windows must yield realistic scales in (0.05, 0.9) and never
exactly `1.0`. On the full extracted set the corrected scale is median
~0.23 (min 0.08 / max 0.43), with zero `1.0`s.
