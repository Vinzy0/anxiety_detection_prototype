# Symptom Monitor Prototype

A real-time webcam-based prototype that detects physical anxiety symptoms using computer vision and MediaPipe landmark tracking. Built as a thesis project exploring affective computing for anxiety symptom recognition.

> **Disclaimer:** This tool is a research prototype only. It does not diagnose anxiety, any mental health condition, or any medical condition whatsoever. It is not a substitute for professional medical or psychological advice. Do not use it to make health decisions.

---

## Overview

The program opens your webcam and tracks five physical symptoms commonly associated with anxiety:

- Rapid blinking
- Lip compression (clenched jaw)
- Hand tremors
- Body restlessness (wrist fidgeting)
- Rapid breathing (elevated breathing rate)

A coping tip is always displayed and updates based on which symptoms are currently active.

---

## Requirements

- Windows 10/11
- Python 3.11
- A working webcam

---

## Setup & Installation

**1. Clone or download the project**

**2. Create and activate a virtual environment**
```
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```
pip install opencv-python mediapipe numpy
```

**4. Run the program**
```
cd anxiety_detection
python main.py
```

On first run, three MediaPipe model files (~7MB each) will download automatically:
- `face_landmarker.task`
- `hand_landmarker.task`
- `pose_landmarker_lite.task`

Press **Q** to quit.

---

## How to Get the Most Accurate Results

- **Sit centered in frame**, upper body visible, face forward
- **Good lighting** — avoid backlighting (don't sit in front of a window)
- **Stay within arm's reach of the camera** — too far and landmark confidence drops
- **Wait ~15 seconds** after starting before trusting the readings — breathing detection needs a 10-second buffer to warm up, restlessness needs 5 seconds
- **Keep your background plain if possible** — busy backgrounds reduce pose tracking reliability
- **Don't cover your face** — eye and mouth detection require a clear face mesh
- The system is designed for a **single person** seated in front of a webcam, not groups or moving around

---

## Project Structure

```
anxiety_detection/
├── main.py                      # entry point — runs the camera loop (background thread) and launches settings panel
├── coping_tips.py               # coping tip strings and symptom-to-tip selection logic
│
├── detection/
│   ├── eye_detection.py         # rapid blinking via Eye Aspect Ratio (EAR)
│   ├── mouth_detection.py       # lip compression via Mouth Aspect Ratio (MAR)
│   ├── hand_detection.py        # hand tremor via per-hand wrist jitter tracking
│   ├── body_detection.py        # restlessness (wrist reversal rate) + breathing (FFT on shoulder Y)
│   └── symptom_checker.py       # combines all 5 flags, triggers coping tip response when >= 2 are active
│
└── ui/
    ├── display.py               # draws the sidebar panel next to the video feed
    └── settings_panel.py        # live threshold adjustment window (tkinter, separate window)
```

---

## How Each Signal Works

| Symptom | Landmark(s) | Method | Flagged When |
|---|---|---|---|
| **Rapid Blinking** | Eye corners + lids (face mesh) | Eye Aspect Ratio (EAR) — ratio of vertical to horizontal eye opening. Counts blink events over a 10s sliding window | ≥ 5 blinks in 10 seconds |
| **Lip Compression** | Mouth corners + lips (face mesh) | Mouth Aspect Ratio (MAR) — ratio of vertical to horizontal mouth opening. Sustained low MAR = clenched jaw | MAR < 0.10 for 15+ consecutive frames |
| **Hand Tremors** | Wrist (hand landmarks) | Tracks left and right wrists independently using MediaPipe handedness labels. Mean pixel displacement per frame over a 10-frame buffer | Mean displacement > 8.0 px/frame |
| **Restlessness** | Wrists + shoulders (pose) | Wrist positions relative to shoulder midpoint (cancels body sways). Counts direction reversals per second, but only for small movements (3–30px) — filters out intentional reaches | > 1.5 reversals/second over 5s |
| **Breathing Rate** | Shoulder midpoint (pose) | Shoulder Y-position buffered over 10s, smoothed with a 7-frame moving average, then FFT to find dominant oscillation frequency. Only looks in the 0.1–1.0 Hz band (6–60 bpm) | Dominant frequency ≥ 0.4 Hz (24 bpm) |

---

## Response Logic

`symptom_checker.py` triggers a coping tip response when **2 or more symptoms are active simultaneously**. A single symptom alone is not enough — this reduces false positives from things like blinking in bright light or natural hand movement.

A coping tip is always shown and is selected based on whichever symptom is considered most actionable (breathing → breathing tip, restlessness → grounding, etc.). When no symptoms are active, a default tip is shown.

---

## Live Settings Panel

A separate **Detection Thresholds** window opens alongside the camera feed. It lets you tune every detection parameter in real time without restarting the program.

- **Changes take effect immediately** on the next processed frame
- **Not persistent** — closing and reopening the program resets everything to the hardcoded defaults
- Drag any slider or **type a value directly** into the number field and press Enter
- **Reset All to Defaults** button restores every slider at once
- The window is resizable and scrollable with the mouse wheel

### Adjustable parameters

| Section | Parameter | What it controls |
|---|---|---|
| Eye | EAR Threshold | How closed the eye must be to count as a blink |
| Eye | Blink Rate / 10s | How many blinks in the window triggers the flag |
| Eye | Blink Window (s) | How many seconds of history to count blinks over |
| Mouth | MAR Threshold | How compressed lips must be |
| Mouth | Sustain Frames | How many consecutive frames of compression before flagging |
| Hand | Jitter Threshold | Mean wrist displacement (px/frame) that flags tremor |
| Body | Restlessness | Direction reversals per second to flag restlessness |
| Body | Rest. Noise Floor | Movements below this px count are ignored as tracking noise |
| Body | Rest. Max Delta | Movements above this px count are ignored as intentional |
| Body | Breathing (Hz) | Dominant breathing frequency that flags rapid breathing |
| Body | Breathing Amp. Floor | Minimum FFT signal strength before breathing rate is trusted |
| Alert | Symptoms Required | How many symptoms must be active to trigger a response |

---

## Known Limitations

- **Breathing and restlessness have warm-up periods** (10s and 5s respectively) before they produce any reading
- **Pose landmarks are noisy** — the lite model has 3–5px of frame-to-frame jitter, which is why the breathing signal is smoothed before FFT
- **Restlessness only tracks wrists** — leg bouncing and foot tapping are invisible to the camera
- **Requires consistent lighting and framing** — results degrade significantly in poor conditions
- **Single person only** — multi-person scenes are not supported

---

## Phase Progress

| Phase | Description | Status |
|---|---|---|
| 0 | Environment setup | Done |
| 1 | Webcam + face landmarks | Done |
| 2 | Eye / blink detection | Done |
| 3 | Mouth / lip compression | Done |
| 4 | Hand tremor detection | Done |
| 5 | Body restlessness + breathing | Done |
| 6 | Symptom checker + coping tips | Done |
| 7 | UI panel | Done |
| 8 | Testing, calibration & live settings panel | Done |
