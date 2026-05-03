"""
Settings Panel — Anxiety Detection Prototype

Live threshold adjustment window. Changes take effect immediately on the next
processed frame. Nothing is saved — closing and reopening the program resets
everything back to the defaults coded in each detection module.

Active detectors: Hand Tremor, Breathing
Archived detectors: Eye (EAR), Mouth (MAR), Restlessness — see detection/archived/
"""

import tkinter as tk
import threading

import detection.hand_detection  as hand_mod
import detection.body_detection  as body_mod
import detection.symptom_checker as symptom_mod

_settings_lock = threading.Lock()

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#0f0e17"
CARD_BG   = "#1a1929"
ENTRY_BG  = "#232235"
TROUGH    = "#45436a"
SEP       = "#2a293a"
TEXT      = "#fffffe"
SUBTEXT   = "#6e6d85"

COLORS = {
    "hand":  "#7f5af0",
    "body":  "#3da9fc",
    "alert": "#f25f4c",
}

# ── Defaults (mirrors each module's hardcoded constant) ───────────────────────
DEFAULTS = {
    # Hand (FFT-based tremor detection)
    "min_tremor_amp":         50.0,
    "tremor_rel_power":       0.35,
    "tremor_sustained_ratio": 0.5,
    # Body — breathing only (restlessness disabled)
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

    def _build_window(self):
        self.root.title("Detection Settings")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.attributes("-topmost", True)
        self.root.minsize(360, 300)
        self.root.geometry("380x420")

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
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

        def _on_canvas_resize(event):
            canvas.itemconfig(content_id, width=event.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        def _on_content_resize(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        content.bind("<Configure>", _on_content_resize)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Hand ──────────────────────────────────────────────────────────────
        self._section(content, "HAND TREMOR", "hand")
        self._slider(content, "min_tremor_amp", "Min Tremor Amplitude", "hand",
                     1.0, 200.0, 1.0, "FFT peak amplitude floor  —  higher = less sensitive",
                     lambda v: setattr(hand_mod, "MIN_TREMOR_AMP", float(v)))
        self._slider(content, "tremor_rel_power", "Relative Power Threshold", "hand",
                     0.05, 0.80, 0.05, "Fraction of total spectral power in the tremor band",
                     lambda v: setattr(hand_mod, "TREMOR_RELATIVE_POWER_THRESHOLD", float(v)))
        self._slider(content, "tremor_sustained_ratio", "Sustained Ratio", "hand",
                     0.1, 1.0, 0.05, "Fraction of recent windows that must show tremor to flag",
                     lambda v: setattr(hand_mod, "TREMOR_SUSTAINED_RATIO", float(v)))

        self._divider(content)

        # ── Body — breathing only ─────────────────────────────────────────────
        self._section(content, "BREATHING", "body")
        self._slider(content, "breathing_threshold", "Breathing (Hz)", "body",
                     0.2, 0.8, 0.05, "Lower = flags slower breathing rates",
                     lambda v: setattr(body_mod, "BREATHING_THRESHOLD", float(v)))
        self._slider(content, "min_breathing_amp", "Breathing Amp. Floor", "body",
                     0.5, 10.0, 0.5, "Min FFT amplitude — higher filters weak signals",
                     lambda v: setattr(body_mod, "MIN_BREATHING_AMP", float(v)))

        self._divider(content)

        # ── Alert ─────────────────────────────────────────────────────────────
        self._section(content, "ALERT SENSITIVITY", "alert")
        self._slider(content, "symptoms_required", "Symptoms Required", "alert",
                     1, 3, 1, "How many symptoms must be active to trigger a response",
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
            v = max(from_, min(to, v))
            return round(round(v / resolution) * resolution, 10)

        card = tk.Frame(parent, bg=CARD_BG, padx=12, pady=8)
        card.pack(fill="x", pady=2)

        top = tk.Frame(card, bg=CARD_BG)
        top.pack(fill="x")
        tk.Label(top, text=label, font=("Segoe UI", 9),
                 bg=CARD_BG, fg=TEXT, anchor="w").pack(side="left")

        val_var = tk.StringVar(value=fmt(default))
        entry = tk.Entry(
            top, textvariable=val_var, width=7,
            font=("Segoe UI", 9, "bold"), bg=ENTRY_BG, fg=accent,
            insertbackground=accent, relief="flat", bd=0,
            highlightthickness=1, highlightcolor=accent,
            highlightbackground=SEP, justify="right",
        )
        entry.pack(side="right")

        scale = tk.Scale(
            card, from_=from_, to=to, resolution=resolution,
            orient=tk.HORIZONTAL, showvalue=False,
            bg=CARD_BG, fg=TEXT, troughcolor=TROUGH,
            activebackground=accent, highlightthickness=0,
            bd=0, sliderrelief="flat", width=10,
        )
        scale.set(default)
        scale.pack(fill="x", pady=(4, 2))

        tk.Label(card, text=hint, font=("Segoe UI", 7),
                 bg=CARD_BG, fg=SUBTEXT, anchor="w").pack(fill="x")

        def on_slide(v):
            val_var.set(fmt(float(v)))
            with _settings_lock:
                on_change(v)
        scale.config(command=on_slide)

        def apply_entry(event=None):
            try:
                v = snap(float(val_var.get()))
                scale.set(v)
                val_var.set(fmt(v))
                with _settings_lock:
                    on_change(v)
            except ValueError:
                val_var.set(fmt(scale.get()))
        entry.bind("<Return>",   apply_entry)
        entry.bind("<FocusOut>", apply_entry)

        self.sliders[key] = scale

    def _divider(self, parent):
        tk.Frame(parent, bg=SEP, height=1).pack(fill="x", pady=(8, 0))

    def _reset_all(self):
        with _settings_lock:
            hand_mod.MIN_TREMOR_AMP                  = DEFAULTS["min_tremor_amp"]
            hand_mod.TREMOR_RELATIVE_POWER_THRESHOLD = DEFAULTS["tremor_rel_power"]
            hand_mod.TREMOR_SUSTAINED_RATIO          = DEFAULTS["tremor_sustained_ratio"]
            body_mod.BREATHING_THRESHOLD             = DEFAULTS["breathing_threshold"]
            body_mod.MIN_BREATHING_AMP               = DEFAULTS["min_breathing_amp"]
            symptom_mod.SYMPTOMS_REQUIRED            = DEFAULTS["symptoms_required"]

        for key, scale in self.sliders.items():
            scale.set(DEFAULTS[key])


def launch_settings_panel():
    """Create and run the settings window. Must be called from the main thread."""
    root = tk.Tk()
    SettingsPanel(root)
    root.mainloop()
