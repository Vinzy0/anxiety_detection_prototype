import csv
import os
import time
from datetime import datetime

LOG_FILE = 'anxiety_log.csv'
LOG_INTERVAL = 30  # seconds between periodic entries


class AnxietyLogger:
    def __init__(self):
        self._prev_state = None
        self._state_start = time.time()
        self._last_log_time = None

        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'anxiety_detected', 'active_symptoms', 'duration_seconds'])

    def update(self, anxiety_detected: bool, active_symptoms: list):
        now = time.time()
        state_changed = (self._prev_state is None) or (anxiety_detected != self._prev_state)
        periodic_due = (self._last_log_time is None) or (now - self._last_log_time >= LOG_INTERVAL)

        if state_changed:
            self._state_start = now

        if state_changed or periodic_due:
            duration = round(now - self._state_start, 1)
            self._write_row(anxiety_detected, active_symptoms, duration)
            self._last_log_time = now

        self._prev_state = anxiety_detected

    def _write_row(self, anxiety_detected: bool, active_symptoms: list, duration_seconds: float):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        symptoms_str = ','.join(active_symptoms) if active_symptoms else ''
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, anxiety_detected, symptoms_str, duration_seconds])
