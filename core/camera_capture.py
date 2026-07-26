"""
camera_capture.py

Thin wrapper around OpenCV's VideoCapture for grabbing webcam frames
on Windows. Keeping this isolated makes it easy to swap camera index,
resolution, or backend (e.g. cv2.CAP_DSHOW) without touching the rest
of the pipeline.
"""

import cv2


class CameraCapture:
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        # CAP_DSHOW is the recommended backend on Windows -- it avoids
        # the slow/blank-frame startup issues that plain cv2.VideoCapture()
        # sometimes has on Windows with certain webcam drivers.
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check that no other app is using "
                "it and that camera permissions are enabled in Windows "
                "Settings > Privacy & Security > Camera."
            )

    def read_frame(self):
        """Returns (success, frame) where frame is a BGR numpy array."""
        success, frame = self.cap.read()
        if not success:
            return False, None
        return True, frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()
