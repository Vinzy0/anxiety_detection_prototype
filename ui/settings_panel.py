"""
Settings Panel — Phase 8 of Anxiety Detection Prototype

Live threshold adjustment window. Changes take effect immediately on the next
processed frame. Nothing is saved — closing and reopening the program resets
everything back to the defaults coded in each detection module.
"""

import tkinter as tk

import detection.eye_detection   as eye_mod
import detection.mouth_detection as mouth_mod
import detection.hand_detection  as hand_mod
import detection.body_detection  as body_mod
import detection.symptom_checker as symptom_mod

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#0f0e17"   # window background
CARD_BG   = "#1a1929"   # slider card background
ENTRY_BG  = "#232235"   # entry field background (slightly lighter than card)
TROUGH    = "#45436a"   # slider trough — clearly visible against card
SEP       = "#2a293a"   # thin divider lines
TEXT      = "#fffffe"
SUBTEXT   = "#6e6d85"

COLORS = {
    "eye":   "#2cb67d",
    "mouth": "#ff8906",
    "hand":  "#7f5af0",
    "body":  "#3da9fc",
    "alert": "#f25f4c",
}

# ── Defaults (mirrors each module's hardcoded constant) ───────────────────────
DEFAULTS = {
    # Eye
    "ear_threshold":          0.22,
    "blink_rate_threshold":   5,
    "time_window":            10,
    # Mouth
    "mar_threshold":          0.10,
    "compression_frames":     15,
    # Hand
    "jitter_threshold":       8.0,
    # Body
    "restlessness_threshold": 1.5,
    "rest_min_delta":         3.0,
    "rest_max_delta":         30.0,
    "breathing_threshold":    0.4,
    "min_breathing_amp":      2.0,
    # Alert
    "symptoms_required":      2,
}


class SettingsPanel:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.sliders: dict[str, tk.Scale] = {}
        self._build_window()
        self._build_ui()

    # ── Window chrome ──────────────────────────────────────────────────────────

    def _build_window(self):
        self.root.title("Detection Settings")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.attributes("-topmost", True)
        self.root.minsize(360, 300)
        self.root.geometry("380x600")

    # ── Full UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header (fixed, outside scroll area) ───────────────────────────────
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(16, 4))
        tk.Label(hdr, text="Detection Thresholds",
                 font=("Segoe UI", 12, "bold"), bg=BG, fg=TEXT, anchor="w").pack(side="left")
        tk.Label(hdr, text="session only",
                 font=("Segoe UI", 8), bg=BG, fg=SUBTEXT, anchor="e").pack(side="right", pady=(4, 0))
        tk.Frame(self.root, bg=SEP, height=1).pack(fill="x", padx=20, pady=(0, 4))

        # ── Scrollable area ───────────────────────────────────────────────────
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(outer, orient="vertical", bg=BG,
                                 troughcolor=BG, bd=0, highlightthickness=0,
                                 activebackground=TROUGH)
        scrollbar.pack(side="right", fill="y")

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0,
                           yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=canvas.yview)

        content = tk.Frame(canvas, bg=BG, padx=14)
        content_id = canvas.create_window((0, 0), window=content, anchor="nw")

        # Keep the inner frame the same width as the canvas
        def _on_canvas_resize(event):
            canvas.itemconfig(content_id, width=event.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        # Update scroll region whenever content size changes
        def _on_content_resize(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        content.bind("<Configure>", _on_content_resize)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Eye ───────────────────────────────────────────────────────────────
        self._section(content, "EYE DETECTION", "eye")
        self._slider(content, "ear_threshold", "EAR Threshold", "eye",
                     0.10, 0.40, 0.01, "Lower  →  more sensitive to closed eyes",
                     lambda v: setattr(eye_mod, "EAR_THRESHOLD", float(v)))
        self._slider(content, "blink_rate_threshold", "Blink Rate / 10s", "eye",
                     1, 15, 1, "Higher  →  requires more blinks to flag",
                     lambda v: setattr(eye_mod, "BLINK_RATE_THRESHOLD", int(float(v))))
        self._slider(content, "time_window", "Blink Window (s)", "eye",
                     5, 30, 1, "Seconds to count blinks over",
                     lambda v: setattr(eye_mod, "TIME_WINDOW", int(float(v))))

        self._divider(content)

        # ── Mouth ─────────────────────────────────────────────────────────────
        self._section(content, "MOUTH DETECTION", "mouth")
        self._slider(content, "mar_threshold", "MAR Threshold", "mouth",
                     0.05, 0.30, 0.01, "Lower  →  tighter lip compression needed",
                     lambda v: setattr(mouth_mod, "MAR_THRESHOLD", float(v)))
        self._slider(content, "compression_frames", "Sustain Frames", "mouth",
                     5, 60, 1, "Frames lips must stay compressed before flagging",
                     lambda v: setattr(mouth_mod, "COMPRESSION_FRAME_THRESHOLD", int(float(v))))

        self._divider(content)

        # ── Hand ──────────────────────────────────────────────────────────────
        self._section(content, "HAND DETECTION", "hand")
        self._slider(content, "jitter_threshold", "Jitter Threshold", "hand",
                     2.0, 30.0, 0.5, "px/frame  —  higher  →  less sensitive",
                     lambda v: setattr(hand_mod, "JITTER_THRESHOLD", float(v)))

        self._divider(content)

        # ── Body ──────────────────────────────────────────────────────────────
        self._section(content, "BODY DETECTION", "body")
        self._slider(content, "restlessness_threshold", "Restlessness", "body",
                     0.5, 5.0, 0.1, "Reversals/sec  —  higher  →  less sensitive",
                     lambda v: setattr(body_mod, "RESTLESSNESS_THRESHOLD", float(v)))
        self._slider(content, "rest_min_delta", "Rest. Noise Floor (px)", "body",
                     1.0, 15.0, 0.5, "Movements below this are ignored as tracking noise",
                     lambda v: setattr(body_mod, "RESTLESSNESS_MIN_DELTA", float(v)))
        self._slider(content, "rest_max_delta", "Rest. Max Delta (px)", "body",
                     10.0, 80.0, 1.0, "Movements above this are ignored as intentional",
                     lambda v: setattr(body_mod, "RESTLESSNESS_MAX_DELTA", float(v)))
        self._slider(content, "breathing_threshold", "Breathing (Hz)", "body",
                     0.2, 0.8, 0.05, "Lower  →  flags slower breathing rates",
                     lambda v: setattr(body_mod, "BREATHING_THRESHOLD", float(v)))
        self._slider(content, "min_breathing_amp", "Breathing Amp. Floor", "body",
                     0.5, 10.0, 0.5, "Min FFT amplitude — higher filters weak signals",
                     lambda v: setattr(body_mod, "MIN_BREATHING_AMP", float(v)))

        self._divider(content)

        # ── Alert ─────────────────────────────────────────────────────────────
        self._section(content, "ALERT SENSITIVITY", "alert")
        self._slider(content, "symptoms_required", "Symptoms Required", "alert",
                     1, 5, 1, "How many symptoms must be active to trigger a response",
                     lambda v: setattr(symptom_mod, "SYMPTOMS_REQUIRED", int(float(v))))

        # Reset button
        tk.Frame(content, bg=BG, height=6).pack()
        tk.Button(
            content, text="↺   Reset All to Defaults",
            font=("Segoe UI", 9), bg=SEP, fg=SUBTEXT,
            activebackground="#2e2d45", activeforeground=TEXT,
            relief="flat", bd=0, padx=0, pady=10, cursor="hand2",
            command=self._reset_all,
        ).pack(fill="x")
        tk.Frame(content, bg=BG, height=12).pack()

    # ── Widget helpers ─────────────────────────────────────────────────────────

    def _section(self, parent, title: str, color_key: str):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(10, 3))
        tk.Frame(row, bg=COLORS[color_key], width=3).pack(side="left", fill="y", padx=(0, 8))
        tk.Label(row, text=title, font=("Segoe UI", 7, "bold"),
                 bg=BG, fg=COLORS[color_key], anchor="w").pack(side="left")

    def _slider(self, parent, key: str, label: str, color_key: str,
                from_: float, to: float, resolution: float,
                hint: str, on_change):
        accent  = COLORS[color_key]
        is_int  = (resolution == 1)
        default = DEFAULTS[key]

        def fmt(v: float) -> str:
            if is_int:            return str(int(v))
            if resolution < 0.05: return f"{v:.2f}"
            return f"{v:.1f}"

        def snap(v: float) -> float:
            """Snap v to the nearest resolution step within [from_, to]."""
            v = max(from_, min(to, v))
            return round(round(v / resolution) * resolution, 10)

        card = tk.Frame(parent, bg=CARD_BG, padx=12, pady=8)
        card.pack(fill="x", pady=2)

        # ── Top row: label + editable value ───────────────────────────────────
        top = tk.Frame(card, bg=CARD_BG)
        top.pack(fill="x")
        tk.Label(top, text=label, font=("Segoe UI", 9),
                 bg=CARD_BG, fg=TEXT, anchor="w").pack(side="left")

        val_var = tk.StringVar(value=fmt(default))

        entry = tk.Entry(
            top,
            textvariable=val_var,
            width=7,
            font=("Segoe UI", 9, "bold"),
            bg=ENTRY_BG, fg=accent,
            insertbackground=accent,
            relief="flat", bd=0,
            highlightthickness=1,
            highlightcolor=accent,
            highlightbackground=SEP,
            justify="right",
        )
        entry.pack(side="right")

        # ── Slider ────────────────────────────────────────────────────────────
        scale = tk.Scale(
            card,
            from_=from_, to=to, resolution=resolution,
            orient=tk.HORIZONTAL, showvalue=False,
            bg=CARD_BG, fg=TEXT,
            troughcolor=TROUGH,
            activebackground=accent,
            highlightthickness=0, bd=0, sliderrelief="flat",
            width=10,
        )
        scale.set(default)
        scale.pack(fill="x", pady=(4, 2))

        # ── Hint ──────────────────────────────────────────────────────────────
        tk.Label(card, text=hint, font=("Segoe UI", 7),
                 bg=CARD_BG, fg=SUBTEXT, anchor="w").pack(fill="x")

        # ── Wiring: slider → entry, entry → slider ────────────────────────────
        def on_slide(v):
            val_var.set(fmt(float(v)))
            on_change(v)

        scale.config(command=on_slide)

        def apply_entry(event=None):
            try:
                v = snap(float(val_var.get()))
                scale.set(v)
                val_var.set(fmt(v))
                on_change(v)
            except ValueError:
                val_var.set(fmt(scale.get()))  # revert to last good value

        entry.bind("<Return>",   apply_entry)
        entry.bind("<FocusOut>", apply_entry)

        self.sliders[key] = scale

    def _divider(self, parent):
        tk.Frame(parent, bg=SEP, height=1).pack(fill="x", pady=(8, 0))

    # ── Reset ──────────────────────────────────────────────────────────────────

    def _reset_all(self):
        eye_mod.EAR_THRESHOLD                  = DEFAULTS["ear_threshold"]
        eye_mod.BLINK_RATE_THRESHOLD           = DEFAULTS["blink_rate_threshold"]
        eye_mod.TIME_WINDOW                    = DEFAULTS["time_window"]
        mouth_mod.MAR_THRESHOLD                = DEFAULTS["mar_threshold"]
        mouth_mod.COMPRESSION_FRAME_THRESHOLD  = DEFAULTS["compression_frames"]
        hand_mod.JITTER_THRESHOLD              = DEFAULTS["jitter_threshold"]
        body_mod.RESTLESSNESS_THRESHOLD        = DEFAULTS["restlessness_threshold"]
        body_mod.RESTLESSNESS_MIN_DELTA        = DEFAULTS["rest_min_delta"]
        body_mod.RESTLESSNESS_MAX_DELTA        = DEFAULTS["rest_max_delta"]
        body_mod.BREATHING_THRESHOLD           = DEFAULTS["breathing_threshold"]
        body_mod.MIN_BREATHING_AMP             = DEFAULTS["min_breathing_amp"]
        symptom_mod.SYMPTOMS_REQUIRED          = DEFAULTS["symptoms_required"]

        for key, scale in self.sliders.items():
            scale.set(DEFAULTS[key])


# ── Entry point ────────────────────────────────────────────────────────────────

def launch_settings_panel():
    """Create and run the settings window. Must be called from the main thread."""
    root = tk.Tk()
    SettingsPanel(root)
    root.mainloop()
