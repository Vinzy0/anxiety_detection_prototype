# How many symptoms must be flagged simultaneously to trigger the anxiety alert
SYMPTOMS_REQUIRED = 2

# Symptom names for display
SYMPTOM_NAMES = {
    "rapid_blinking":  "Rapid Blinking",
    "lip_compression": "Lip Compression",
    "hand_tremor":     "Hand Tremors",
    "restlessness":    "Body Restlessness",
    "rapid_breathing": "Rapid Breathing",
}


class SymptomChecker:
    def __init__(self):
        self.active_symptoms = []
        self.anxiety_detected = False

    def update(self, eye_flagged, mouth_flagged, hand_flagged, rest_flagged, breath_flagged):
        self.active_symptoms = []

        if eye_flagged:
            self.active_symptoms.append("rapid_blinking")
        if mouth_flagged:
            self.active_symptoms.append("lip_compression")
        if hand_flagged:
            self.active_symptoms.append("hand_tremor")
        if rest_flagged:
            self.active_symptoms.append("restlessness")
        if breath_flagged:
            self.active_symptoms.append("rapid_breathing")

        self.anxiety_detected = len(self.active_symptoms) >= SYMPTOMS_REQUIRED

        return self.anxiety_detected, self.active_symptoms
