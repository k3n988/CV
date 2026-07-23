"""
face_detector.py

Wraps MediaPipe Face Detection to locate a face in a frame and return
a cropped region ready to feed into the expression classifier.

MediaPipe Face Detection is used here (rather than Face Mesh) because
we only need a bounding box for cropping -- Face Mesh's 468 landmarks
are more detail than the expression model needs, and would just add
inference cost.
"""

import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.6):
        self._mp_face_detection = mp.solutions.face_detection
        self.detector = self._mp_face_detection.FaceDetection(
            model_selection=0,  # 0 = short-range model, tuned for faces within ~2m (webcam use case)
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame_bgr):
        """
        Runs detection on a single BGR frame.

        Returns a list of dicts: [{"bbox": (x, y, w, h), "confidence": float}, ...]
        Coordinates are in pixels, clamped to the frame bounds.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(frame_rgb)

        detections = []
        if results.detections:
            h, w, _ = frame_bgr.shape
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                x = max(0, int(box.xmin * w))
                y = max(0, int(box.ymin * h))
                bw = min(w - x, int(box.width * w))
                bh = min(h - y, int(box.height * h))
                confidence = det.score[0] if det.score else 0.0
                detections.append({"bbox": (x, y, bw, bh), "confidence": confidence})

        return detections

    @staticmethod
    def crop_face(frame_bgr, bbox):
        """Crops the face region from a frame given a (x, y, w, h) bbox."""
        x, y, w, h = bbox
        return frame_bgr[y:y + h, x:x + w]

    def close(self):
        self.detector.close()
