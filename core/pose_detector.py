"""
pose_detector.py

Wraps MediaPipe's Tasks API PoseLandmarker to extract simple posture
cues:
- head-down posture (nose landmark significantly below shoulder line)
- stillness (very small landmark movement across frames)

These are intentionally coarse, per the concept paper's scope: they
are supporting signals, not standalone diagnostic measurements.

NOTE: This uses the newer MediaPipe Tasks API (mediapipe.tasks.python),
not the older mp.solutions.pose API, which was removed in recent
MediaPipe releases (0.10.30+).
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from core.model_downloader import ensure_model


class PoseDetector:
    # PoseLandmarker landmark indices we care about (same indexing as
    # the older BlazePose model used by mp.solutions.pose)
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def __init__(self, min_detection_confidence: float = 0.5, history_len: int = 15):
        model_path = ensure_model("pose_landmarker_lite.task")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        # Tasks API returns a list of pose landmark lists (one per detected
        # person) -- take the first detected pose.
        landmarks = result.pose_landmarks[0]
        nose_y = landmarks[self.NOSE].y
        shoulder_y = (landmarks[self.LEFT_SHOULDER].y + landmarks[self.RIGHT_SHOULDER].y) / 2

        # In normalized image coords, y grows downward. If nose is close to
        # or below the shoulder line, that's a strong head-down posture cue.
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
        self.landmarker.close()