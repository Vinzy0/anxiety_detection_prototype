# Module-level constants below are written by the settings panel (main thread)
# and read by the camera loop (daemon thread). Writes are locked via
# settings_panel._settings_lock. Reads are not locked — safe under CPython's GIL
# for simple scalar assignment, but not guaranteed under GIL-free runtimes.

# How many symptoms must be flagged simultaneously to trigger the anxiety alert
SYMPTOMS_REQUIRED = 2

# Symptom names for display
SYMPTOM_NAMES = {
    "hand_tremor":     "Hand Tremors",
    "restlessness":    "Body Restlessness",
    "rapid_breathing": "Rapid Breathing",
}


class SymptomChecker:
    def __init__(self):
        self.active_symptoms = []
        self.anxiety_detected = False

    def update(self, hand_flagged, rest_flagged, breath_flagged):
        self.active_symptoms = []

        if hand_flagged:
            self.active_symptoms.append("hand_tremor")
        if rest_flagged:
            self.active_symptoms.append("restlessness")
        if breath_flagged:
            self.active_symptoms.append("rapid_breathing")

        self.anxiety_detected = len(self.active_symptoms) >= SYMPTOMS_REQUIRED

        return self.anxiety_detected, self.active_symptoms
