"""
face_detector.py

Wraps MediaPipe's Tasks API FaceDetector to locate a face in a frame
and return a cropped region ready to feed into the expression
classifier.

NOTE: This uses the newer MediaPipe Tasks API (mediapipe.tasks.python),
not the older mp.solutions.face_detection API, which was removed in
recent MediaPipe releases (0.10.30+).
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from core.model_downloader import ensure_model


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.6):
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
        Runs detection on a single BGR frame.

        Returns a list of dicts: [{"bbox": (x, y, w, h), "confidence": float}, ...]
        Coordinates are in pixels, clamped to the frame bounds.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        detections = []
        if result.detections:
            h, w, _ = frame_bgr.shape
            for det in result.detections:
                box = det.bounding_box
                x = max(0, box.origin_x)
                y = max(0, box.origin_y)
                bw = min(w - x, box.width)
                bh = min(h - y, box.height)
                confidence = det.categories[0].score if det.categories else 0.0
                detections.append({"bbox": (x, y, bw, bh), "confidence": confidence})

        return detections

    @staticmethod
    def crop_face(frame_bgr, bbox):
        """Crops the face region from a frame given a (x, y, w, h) bbox."""
        x, y, w, h = bbox
        return frame_bgr[y:y + h, x:x + w]

    def close(self):
        self.detector.close()