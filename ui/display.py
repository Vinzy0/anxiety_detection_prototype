import cv2
import numpy as np

PANEL_W = 300   # width of the sidebar in pixels

# ── Color palette (BGR) ───────────────────────────────────────────────────────
BG      = (17,  17,  17)    # panel background
CARD    = (30,  30,  30)    # section card background
SEP     = (46,  46,  46)    # separator lines
WHITE   = (224, 224, 224)   # primary text
MUTED   = (110, 110, 110)   # secondary / label text
TEAL    = (166, 184, 20)    # #14b8a6 — normal / OK accent
RED     = (68,  68,  239)   # #ef4444 — alert
AMBER   = (11,  158, 245)   # #f59e0b — mid-level warning
YELLOW  = (36,  191, 251)   # #fbbf24 — coping tip accent

_SYMPTOMS = [
    ("rapid_blinking",  "Rapid Blinking"),
    ("lip_compression", "Lip Compression"),
    ("hand_tremor",     "Hand Tremors"),
    ("restlessness",    "Body Restlessness"),
    ("rapid_breathing", "Rapid Breathing"),
]


def _text(canvas, msg, x, y, color, scale=0.44, thickness=1):
    cv2.putText(canvas, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _hline(canvas, y, x0, x1):
    cv2.line(canvas, (x0, y), (x1, y), SEP, 1)


def _bar(canvas, x, y, w, value, max_val):
    """Thin progress bar, 6 px tall."""
    h = 6
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD, -1)
    ratio   = min(value / max_val, 1.0) if max_val > 0 else 0.0
    fill_px = int(w * ratio)
    color   = RED if ratio > 0.75 else (AMBER if ratio > 0.45 else TEAL)
    if fill_px > 0:
        cv2.rectangle(canvas, (x, y), (x + fill_px, y + h), color, -1)


def draw_symptom_panel(frame, active_symptoms, anxiety_detected, coping_tip, metrics=None):
    """
    Returns a new canvas: the original video frame on the left,
    a dark status panel on the right — no overlap.
    """
    fh, fw = frame.shape[:2]

    # ── Build canvas ─────────────────────────────────────────────────────────
    canvas = np.full((fh, fw + PANEL_W, 3), BG, dtype=np.uint8)
    canvas[:, :fw] = frame

    # Thin separator line between video and panel
    cv2.line(canvas, (fw, 0), (fw, fh), SEP, 1)

    px  = fw + 14           # left text margin inside panel
    rx  = fw + PANEL_W - 14 # right edge for right-aligned items
    bar_w = PANEL_W - 32

    # ── Header ───────────────────────────────────────────────────────────────
    _text(canvas, "SYMPTOM MONITOR", px, 28, WHITE, scale=0.50, thickness=1)
    accent = RED if anxiety_detected else TEAL
    cv2.line(canvas, (fw + 8, 36), (fw + PANEL_W - 8, 36), accent, 2)

    # ── Symptom rows ─────────────────────────────────────────────────────────
    y = 60
    for key, label in _SYMPTOMS:
        active = key in active_symptoms
        color  = RED if active else TEAL

        # Status dot
        cv2.circle(canvas, (px + 5, y - 5), 5, color, -1 if active else 1, cv2.LINE_AA)

        # Label
        _text(canvas, label, px + 18, y, WHITE if active else MUTED)

        # Right-aligned badge
        badge = "ALERT" if active else "OK"
        bx    = rx - len(badge) * 8
        _text(canvas, badge, bx, y, color, scale=0.38)

        y += 33

    # ── Live metrics ─────────────────────────────────────────────────────────
    if metrics:
        _hline(canvas, y + 6, fw + 8, fw + PANEL_W - 8)
        y += 22
        _text(canvas, "LIVE METRICS", px, y, MUTED, scale=0.38)
        y += 16

        for label, value, max_val in metrics:
            val_str = f"{value:.2f}"
            _text(canvas, label, px, y, MUTED, scale=0.36)
            _text(canvas, val_str, rx - len(val_str) * 7, y, MUTED, scale=0.36)
            y += 13
            _bar(canvas, px, y, bar_w, value, max_val)
            y += 18

    # ── Status section ───────────────────────────────────────────────────────
    _hline(canvas, y + 6, fw + 8, fw + PANEL_W - 8)
    y += 24

    if coping_tip:
        _text(canvas, "Tip:", px, y, YELLOW, scale=0.40)
        y += 18

        # Word-wrap the coping tip
        words, line = coping_tip.split(), ""
        for word in words:
            if len(line) + len(word) + 1 <= 30:
                line += word + " "
            else:
                _text(canvas, line.strip(), px, y, WHITE, scale=0.38)
                y += 16
                line = word + " "
        if line:
            _text(canvas, line.strip(), px, y, WHITE, scale=0.38)

    # ── Footer ───────────────────────────────────────────────────────────────
    _text(canvas, "Press Q to quit", fw + 14, fh - 12, MUTED, scale=0.38)

    return canvas
