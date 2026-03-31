COPING_TIPS = [
    "Take a slow deep breath. Inhale for 4 counts, hold for 4, exhale for 4.",
    "Ground yourself: name 5 things you can see around you.",
    "Relax your shoulders — let them drop away from your ears.",
    "Unclench your jaw and relax your hands.",
    "Try box breathing: breathe in, hold, breathe out, hold — 4 counts each.",
]


def get_tip(active_symptoms):
    if "rapid_breathing" in active_symptoms:
        return COPING_TIPS[0]
    elif "restlessness" in active_symptoms:
        return COPING_TIPS[1]
    elif "hand_tremor" in active_symptoms:
        return COPING_TIPS[3]
    elif "lip_compression" in active_symptoms:
        return COPING_TIPS[2]
    else:
        return COPING_TIPS[4]
