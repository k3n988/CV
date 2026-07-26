"""
face_detector.py

Wraps MediaPipe's Tasks API FaceDetector to get a single face bounding
box per frame. This drives the diagnostic HUD reticle (core/hud_overlay.py)
and supplies the crop used by ExpressionClassifier -- it does not do any
identity/recognition, only "is there a face, and roughly where."

NOTE: Uses the newer MediaPipe Tasks API (mediapipe.tasks.python), same
as pose_detector.py and hand_detector.py, for consistency.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from core.model_downloader import ensure_model


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        model_path = ensure_model("blaze_face_short_range.tflite")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, frame_bgr):
        """
        Returns (x, y, w, h) pixel bbox of the largest detected face in
        frame_bgr's own pixel space, or None if no face is detected.
        If multiple faces are found, the largest (closest) one is used --
        this app is designed around a single user facing the camera.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        if not result.detections:
            return None

        def area(det):
            b = det.bounding_box
            return b.width * b.height

        best = max(result.detections, key=area)
        b = best.bounding_box
        return (b.origin_x, b.origin_y, b.width, b.height)

    def close(self):
        self.detector.close()