"""
hand_detector.py

Wraps MediaPipe's Tasks API HandLandmarker to detect hand and finger
landmarks (21 points per hand). Used purely for the visual skeleton
overlay on the video feed -- it does not feed into the response engine
or expression/posture logic.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from core.model_downloader import ensure_model


class HandDetector:
    def __init__(self, num_hands: int = 2, min_detection_confidence: float = 0.5):
        model_path = ensure_model("hand_landmarker.task")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        """
        Returns a list of landmark lists (one per detected hand, up to
        num_hands), each a list of 21 normalized landmarks (x, y in
        0.0-1.0). Returns an empty list if no hands are detected.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)
        return result.hand_landmarks or []

    def close(self):
        self.landmarker.close()