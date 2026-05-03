import numpy as np

# Module-level constants below are written by the settings panel (main thread)
# and read by the camera loop (daemon thread). Writes are locked via
# settings_panel._settings_lock. Reads are not locked — safe under CPython's GIL
# for simple scalar assignment, but not guaranteed under GIL-free runtimes.

MOUTH_TOP = 0        # outer top of upper lip
MOUTH_BOTTOM = 17    # outer bottom of lower lip
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

MAR_THRESHOLD = 0.10
COMPRESSION_FRAME_THRESHOLD = 15


def calculate_mar(mouth_top, mouth_bottom, mouth_left, mouth_right):
    vertical = np.linalg.norm(np.array(mouth_top) - np.array(mouth_bottom))
    horizontal = np.linalg.norm(np.array(mouth_left) - np.array(mouth_right))

    if horizontal == 0:
        return 0.0

    mar = vertical / horizontal
    return mar


def get_landmark_coords(face_landmarks, idx, frame_w, frame_h):
    lm = face_landmarks[idx]
    return (int(lm.x * frame_w), int(lm.y * frame_h))


class MouthDetector:
    def __init__(self):
        self.compression_frames = 0
        self.flagged = False

    def update(self, face_landmarks, frame_w, frame_h):
        top = get_landmark_coords(face_landmarks, MOUTH_TOP, frame_w, frame_h)
        bottom = get_landmark_coords(face_landmarks, MOUTH_BOTTOM, frame_w, frame_h)
        left = get_landmark_coords(face_landmarks, MOUTH_LEFT, frame_w, frame_h)
        right = get_landmark_coords(face_landmarks, MOUTH_RIGHT, frame_w, frame_h)

        mar = calculate_mar(top, bottom, left, right)

        if mar < MAR_THRESHOLD:
            self.compression_frames += 1
        else:
            self.compression_frames = 0

        self.flagged = self.compression_frames >= COMPRESSION_FRAME_THRESHOLD

        return self.flagged, mar

    def reset(self):
        self.compression_frames = 0
        self.flagged = False
