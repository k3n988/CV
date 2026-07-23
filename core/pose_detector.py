"""
pose_detector.py

Wraps MediaPipe Pose Landmarker to extract simple posture cues:
- head-down posture (nose landmark significantly below shoulder line)
- stillness (very small landmark movement across frames)

These are intentionally coarse, per the concept paper's scope: they
are supporting signals, not standalone diagnostic measurements.
"""

import mediapipe as mp
import cv2
import numpy as np


class PoseDetector:
    # MediaPipe Pose landmark indices we care about
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def __init__(self, min_detection_confidence: float = 0.5, history_len: int = 15):
        self._mp_pose = mp.solutions.pose
        self.pose = self._mp_pose.Pose(
            model_complexity=0,  # lite model -- fast enough for real-time webcam use
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        # Rolling history of nose position for stillness calculation
        self._nose_history = []
        self._history_len = history_len

    def detect(self, frame_bgr):
        """
        Returns a dict of posture cues, or None if no pose detected:
        {
            "head_down": bool,
            "still": bool,
            "landmarks": raw mediapipe landmarks (for optional drawing)
        }
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        nose_y = landmarks[self.NOSE].y
        shoulder_y = (landmarks[self.LEFT_SHOULDER].y + landmarks[self.RIGHT_SHOULDER].y) / 2

        # In normalized image coords, y grows downward. If nose is close to
        # or below the shoulder line, that's a strong head-down posture cue.
        shoulder_span = abs(landmarks[self.LEFT_SHOULDER].y - landmarks[self.RIGHT_SHOULDER].y) + 1e-6
        head_down = (nose_y - shoulder_y) > -0.05  # threshold tuned loosely; adjust after testing

        # Stillness: track nose (x, y) over recent frames, flag low movement variance
        self._nose_history.append((landmarks[self.NOSE].x, landmarks[self.NOSE].y))
        if len(self._nose_history) > self._history_len:
            self._nose_history.pop(0)

        still = False
        if len(self._nose_history) == self._history_len:
            arr = np.array(self._nose_history)
            movement = np.std(arr, axis=0).sum()
            still = movement < 0.003  # tuned loosely; adjust after real-world testing

        return {
            "head_down": head_down,
            "still": still,
            "landmarks": landmarks,
        }

    def close(self):
        self.pose.close()
