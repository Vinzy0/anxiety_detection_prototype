"""
Eye Detection Module — Phase 2 of Anxiety Detection Prototype

This module detects rapid blinking, a common anxiety symptom. It works by calculating
the Eye Aspect Ratio (EAR) for each eye using facial landmark coordinates from MediaPipe.

EAR Formula:
    EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)

    Where:
    - p1, p4 are the horizontal points (left/right corners of eye)
    - p2, p3, p5, p6 are the vertical points (upper/lower eyelid)

When the eye is open, EAR is typically 0.25–0.30. When closed (blinking), EAR approaches 0.
We track how many times EAR drops below a threshold within a time window — if it happens
too often, we flag it as rapid blinking (an anxiety indicator).
"""

import numpy as np
from collections import deque
import time

# =============================================================================
# LANDMARK INDICES
# =============================================================================
# MediaPipe Face Landmarker provides 478 landmarks on the face. These indices
# correspond to the specific dots that outline each eye. We use 6 points per eye
# to calculate EAR — the same standard used in the affective computing literature.
#
# LEFT_EYE: [362, 385, 387, 263, 373, 380]
#   Landmark 362 = left corner (outer)
#   Landmark 385 = upper left
#   Landmark 387 = upper right
#   Landmark 263 = right corner (inner)
#   Landmark 373 = lower right
#   Landmark 380 = lower left
#
# RIGHT_EYE: [33, 160, 158, 133, 153, 144]
#   Mirror of left eye indices

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# =============================================================================
# THRESHOLDS
# =============================================================================
# EAR_THRESHOLD: If EAR drops below this value, the eye is considered closed.
#   Typical open eye EAR: 0.25–0.30. Blinking drops it near 0.
#   We use 0.22 as a balance between sensitivity and false positive rate.

EAR_THRESHOLD = 0.22

# BLINK_RATE_THRESHOLD: How many blinks must occur within the time window
#   to trigger the "rapid blinking" flag. 5 blinks in 10 seconds is
#   above the normal human blink rate (~15-20 blinks per minute = 2.5-3.3 per 10s)

BLINK_RATE_THRESHOLD = 5

# TIME_WINDOW: The sliding window size in seconds. We only count blinks that
#   happened within this many seconds from now. Older blinks are discarded.

TIME_WINDOW = 10


def calculate_ear(eye_landmarks):
    """
    Calculates the Eye Aspect Ratio (EAR) for a single eye.

    Args:
        eye_landmarks: List of 6 (x, y) pixel coordinate tuples, ordered as:
                       [left_corner, upper_left, upper_right, right_corner, lower_right, lower_left]

    Returns:
        float: The EAR value. Higher = more open eye. ~0.25-0.30 when open, ~0 when closed.

    The formula measures how tall the eye opening is relative to how wide it is.
    A round eye (open) has significant vertical height. A squeezed/shut eye has none.
    """
    # Get the two vertical distances (top to bottom of the eyelid on each side)
    # These are the "height" of the eye opening
    vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))

    # Get the horizontal distance (width of the eye opening — corner to corner)
    horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))

    # Guard against division by zero (would happen if eye is completely closed
    # and horizontal distance becomes 0)
    if horizontal == 0:
        return 0.0

    # EAR = average vertical / horizontal
    # We sum both vertical distances and divide by (2 * horizontal) as per the
    # standard EAR formula from the literature (Soukupová & Čech, 2016)
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def get_eye_landmarks(face_landmarks, indices, frame_w, frame_h):
    """
    Extracts pixel coordinates for specific eye landmarks from the MediaPipe face mesh.

    MediaPipe returns landmark coordinates as normalized values (0.0 to 1.0) relative
    to the image dimensions. This function converts them to absolute pixel coordinates.

    Args:
        face_landmarks: The face_landmarks object from MediaPipe's FaceLandmarker result.
                        In the new tasks.vision API, this is a list of NormalizedLandmark objects
                        where each landmark has .x, .y, .z attributes. Access via face_landmarks[idx].
        indices: List of landmark indices to extract (e.g., LEFT_EYE or RIGHT_EYE)
        frame_w: Frame width in pixels (used to convert normalized x to pixel x)
        frame_h: Frame height in pixels (used to convert normalized y to pixel y)

    Returns:
        List of (x, y) tuples in pixel coordinates, ordered according to the indices list.
    """
    landmarks = []
    for idx in indices:
        # In the new MediaPipe tasks.vision API, face_landmarks is a list of
        # NormalizedLandmark objects. Each landmark has .x, .y, .z attributes.
        # We index directly into the list to get each landmark.
        lm = face_landmarks[idx]
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        landmarks.append((x, y))
    return landmarks


class EyeDetector:
    """
    Tracks eye state frame-by-frame to detect rapid blinking.

    This class maintains a sliding window of blink timestamps. Each time the EAR
    drops below the threshold, we count it as a blink (but only if the eye wasn't
    already closed — this avoids counting one blink as multiple frames of closure).

    Usage:
        detector = EyeDetector()
        # In your main loop, for each frame with a detected face:
        flagged, ear, blink_count = detector.update(face_landmarks, frame_w, frame_h)
    """

    def __init__(self):
        """
        Initializes the EyeDetector with default state.

        Instance variables:
            blink_timestamps: A deque (double-ended queue) storing the time.time()
                              of each detected blink. We pop from the left when
                              blinks fall outside our TIME_WINDOW.
            eye_closed:       Boolean tracking whether the eye is currently below
                              the EAR threshold. Used to detect the transition
                              from open→closed (which marks one blink), not just
                              the sustained closed state.
            blink_count:      Number of blinks detected in the current time window.
            flagged:          True if blink_count >= BLINK_RATE_THRESHOLD.
        """
        self.blink_timestamps = deque()
        self.eye_closed = False
        self.blink_count = 0
        self.flagged = False

    def update(self, face_landmarks, frame_w, frame_h):
        """
        Analyzes one frame for eye blink detection.

        Call this every frame, passing in the face landmarks from MediaPipe.

        Args:
            face_landmarks: The face_landmarks object from MediaPipe's FaceLandmarker.
                            We use face_landmarks.landmark[idx] to access individual points.
            frame_w:        Frame width in pixels (for coordinate conversion)
            frame_h:        Frame height in pixels (for coordinate conversion)

        Returns:
            tuple: (flagged, ear, blink_count)
                - flagged:     True if rapid blinking is currently detected
                - ear:         Current average Eye Aspect Ratio across both eyes
                - blink_count: Number of blinks in the last TIME_WINDOW seconds
        """
        # Extract the 6 landmark coordinates for each eye and convert to pixels
        left_eye = get_eye_landmarks(face_landmarks, LEFT_EYE, frame_w, frame_h)
        right_eye = get_eye_landmarks(face_landmarks, RIGHT_EYE, frame_w, frame_h)

        # Calculate EAR for each eye individually
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # Average the two EAR values — gives us a more robust measurement
        # (either eye blinking will register, and it smooths out individual variance)
        avg_ear = (left_ear + right_ear) / 2.0

        # -------------------------------------------------------------------------
        # BLINK DETECTION LOGIC
        # -------------------------------------------------------------------------
        # A blink is the moment the eye CLOSES. We detect this by watching for
        # the transition from EAR above threshold → EAR below threshold.
        #
        # If we just checked "is EAR below threshold?" every frame, we'd count
        # one blink as ~3-5 frames of closure. By tracking eye_closed state,
        # we only count when the eye RE-OPENS (closed=True → closed=False).
        # -------------------------------------------------------------------------

        if avg_ear < EAR_THRESHOLD:
            # Eye is currently closed
            if not self.eye_closed:
                # This is a NEW blink — eye was open last frame, but closed now.
                # Record the timestamp of this blink.
                self.eye_closed = True
                self.blink_timestamps.append(time.time())
        else:
            # Eye is open
            self.eye_closed = False

        # -------------------------------------------------------------------------
        # SLIDING WINDOW — remove old blinks
        # -------------------------------------------------------------------------
        # We only want to count blinks within the last TIME_WINDOW seconds.
        # Discard any blink timestamps that are older than that window.
        current_time = time.time()
        while self.blink_timestamps and current_time - self.blink_timestamps[0] > TIME_WINDOW:
            self.blink_timestamps.popleft()

        # Count how many blinks remain in the window
        self.blink_count = len(self.blink_timestamps)

        # Flag if we've exceeded the threshold number of blinks in the window
        self.flagged = self.blink_count >= BLINK_RATE_THRESHOLD

        return self.flagged, avg_ear, self.blink_count
