import tkinter as tk
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ui.settings_panel import _show_disclaimer

root = tk.Tk()
root.configure(bg="#0f0e17")
_show_disclaimer(root)
root.destroy()
