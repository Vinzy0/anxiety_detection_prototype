# Symptom Monitor

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D4?style=flat-square&logo=windows&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-blue?style=flat-square&logo=google&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TFLite-Model-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-purple?style=flat-square)
![Thesis](https://img.shields.io/badge/Type-Thesis%20Project-green?style=flat-square)

> **Disclaimer:** This tool is a research prototype only. It does not diagnose anxiety, any mental health condition, or any medical condition whatsoever. It is not a substitute for professional medical or psychological advice. Do not use it to make health decisions.

---

Real-time webcam-based prototype that detects physical anxiety symptoms using computer vision, MediaPipe landmark tracking, and a trained TFLite classification model. Built as a thesis project exploring affective computing for anxiety symptom recognition.

---

## Detected Symptoms

| Symptom | Method | Flagged When |
|---|---|---|
| **Facial Tension** | Trained TFLite binary classifier on 478-point face mesh landmarks + MediaPipe blendshapes | Model probability crosses threshold |
| **Hand Tremors** | Per-hand wrist jitter tracking via MediaPipe handedness labels over a 10-frame buffer | Mean pixel displacement > 8.0 px/frame |
| **Rapid Breathing** | Shoulder Y-position buffered over 10s → 7-frame smoothing → FFT dominant frequency | Dominant frequency ≥ 0.4 Hz (24 bpm) |

An alert triggers when **2 or more symptoms are active simultaneously** — a single symptom alone is not enough to reduce false positives from natural movement or lighting conditions.

A coping tip is always displayed and updates based on whichever symptom is most actionable.

---

## Requirements

- Windows 10 or 11
- Python 3.11
- A working webcam

---

## Setup

**1. Clone the repo**

**2. Create and activate a virtual environment**
```
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```
pip install opencv-python mediapipe numpy tensorflow
```

**4. Run**
```
cd anxiety_detection
python main.py
```

On first run, the face landmarker model (~7MB) downloads automatically from Google's servers.

Press **Q** to quit.

---

## Project Structure

```
anxiety_detection/
├── main.py                      # entry point — camera loop thread + settings panel
├── coping_tips.py               # tip strings and symptom-to-tip selection logic
├── logger.py                    # session logging to anxiety_log.csv
│
├── detection/
│   ├── facial_detection.py      # TFLite model — classifies facial tension from landmarks + blendshapes
│   ├── hand_detection.py        # hand tremor via per-hand wrist jitter
│   ├── body_detection.py        # breathing rate via FFT on shoulder Y movement
│   └── symptom_checker.py       # combines all flags, triggers alert when >= 2 active
│
├── tflite/
│   ├── facial_tension.tflite    # trained binary classifier
│   ├── scaler_mean.npy          # feature scaler mean
│   └── scaler_std.npy           # feature scaler std
│
└── ui/
    ├── display.py               # draws the sidebar panel onto the video feed
    └── settings_panel.py        # live threshold adjustment window (tkinter)
```

---

## Facial Tension Model

The old rule-based EAR/MAR approach (eye aspect ratio, mouth aspect ratio) has been replaced with a trained **TFLite binary classifier**.

The model takes as input:
- Selected face mesh landmark coordinates (eyebrows, eye corners)
- All MediaPipe blendshape scores (jawOpen, browDown, eyeSquint, etc.)
- A `face_detected` flag

Output: `tense` (True/False) + a confidence probability (0.0–1.0) displayed as a bar in the UI.

---

## Session Logging

Every session writes to `anxiety_log.csv`. Rows are written on **state change** (calm → anxious or vice versa) and **every 30 seconds** as a periodic heartbeat.

```
timestamp, anxiety_detected, active_symptoms, duration_seconds
```

---

## Live Settings Panel

A separate **Detection Thresholds** window opens alongside the camera feed for real-time tuning without restarting.

- Changes take effect immediately on the next processed frame
- Not persistent — restarting resets to hardcoded defaults
- Drag sliders or type a value directly and press Enter
- **Reset All to Defaults** button available

| Section | Parameter | What it controls |
|---|---|---|
| Face | Tension Threshold | Model probability cutoff to flag facial tension |
| Hand | Jitter Threshold | Mean wrist displacement (px/frame) to flag tremor |
| Body | Breathing (Hz) | Dominant FFT frequency to flag rapid breathing |
| Body | Breathing Amp. Floor | Minimum signal strength before breathing is trusted |
| Alert | Symptoms Required | How many symptoms must be active to trigger an alert |

---

## Tips for Accurate Results

- Sit centered in frame with your upper body visible, face forward
- Good lighting — avoid backlighting (don't sit in front of a window)
- Stay within arm's reach of the camera
- Wait ~10 seconds after starting — breathing detection needs a buffer to warm up
- Keep your background plain if possible
- Don't cover your face
- Single person only — multi-person scenes are not supported

---

## Known Limitations

- Breathing detection has a ~10s warm-up before it produces any reading
- Pose landmarks have 3–5px of frame-to-frame noise — the breathing signal is smoothed before FFT to compensate
- Restlessness detection (wrist reversal rate) is implemented but currently disabled in the main loop
- Requires consistent lighting and framing — results degrade significantly in poor conditions
- The facial tension model was trained on a limited dataset; performance varies across faces and lighting conditions

---

## Build Status

| Phase | Description | Status |
|---|---|---|
| 0 | Environment setup | Done |
| 1 | Webcam + face landmarks | Done |
| 2 | Eye / blink detection (deprecated) | Done |
| 3 | Mouth / lip compression (deprecated) | Done |
| 4 | Hand tremor detection | Done |
| 5 | Body restlessness + breathing | Done |
| 6 | Symptom checker + coping tips | Done |
| 7 | UI panel | Done |
| 8 | Live settings panel + calibration | Done |
| 9 | Facial tension TFLite model | Done |
| 10 | Session logging | Done |
