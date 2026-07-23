"""
main_window.py

PyQt6 desktop UI: shows the live webcam feed, the transparently-labeled
detected cue, and the current response tier/message. Per the concept
paper's "Transparency Layer", every detected-cue display is labeled as
an estimate, not a certain statement.
"""

import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

from core.camera_capture import CameraCapture
from core.face_detector import FaceDetector
from core.pose_detector import PoseDetector
from core.expression_classifier import ExpressionClassifier
from core.signal_aggregator import SignalAggregator
from core.response_engine import ResponseEngine

# Run inference every INFERENCE_INTERVAL_MS milliseconds, not every frame,
# to keep the UI smooth. The webcam feed itself still updates every frame.
INFERENCE_INTERVAL_MS = 400
UI_REFRESH_INTERVAL_MS = 33  # ~30 fps for the raw camera preview


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EmoSense -- Expression & Behavior Detector (Demo)")
        self.setMinimumSize(760, 640)

        # --- Pipeline components ---
        self.camera = CameraCapture()
        self.face_detector = FaceDetector()
        self.pose_detector = PoseDetector()
        self.expression_classifier = ExpressionClassifier()
        self.aggregator = SignalAggregator(window_seconds=8.0)
        self.response_engine = ResponseEngine()

        self.consent_given = False
        self._build_ui()

        # Two timers: one fast (camera preview), one slower (heavier inference)
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self._update_preview)

        self.inference_timer = QTimer()
        self.inference_timer.timeout.connect(self._run_inference_cycle)

        self._latest_frame = None

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout()

        # Consent gate -- shown before detection starts, per the concept
        # paper's ethics/consent requirement.
        self.consent_label = QLabel(
            "EmoSense will use your webcam to detect visible facial expressions "
            "and posture. This is an estimate of visible cues only -- it cannot "
            "know your actual internal emotional or health state. No video is "
            "stored or transmitted; everything runs on this device, this session only."
        )
        self.consent_label.setWordWrap(True)
        layout.addWidget(self.consent_label)

        self.consent_button = QPushButton("I understand -- start detection")
        self.consent_button.clicked.connect(self._on_consent_given)
        layout.addWidget(self.consent_button)

        # Video feed
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)

        # Transparency label -- always phrased as an estimate, never a
        # certain claim about the person's actual state.
        self.cue_label = QLabel("Detected cue: -- (detection not yet started)")
        self.cue_label.setWordWrap(True)
        layout.addWidget(self.cue_label)

        self.explainer_label = QLabel(
            "Note: this reflects a visible-cue estimate from the camera only, "
            "not a measurement of your actual internal state."
        )
        self.explainer_label.setStyleSheet("color: gray; font-size: 11px;")
        self.explainer_label.setWordWrap(True)
        layout.addWidget(self.explainer_label)

        # Response message
        self.response_label = QLabel("")
        self.response_label.setWordWrap(True)
        self.response_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(self.response_label)

        # Resource list (Tier 3 only, hidden otherwise)
        self.resources_label = QLabel("")
        self.resources_label.setWordWrap(True)
        self.resources_label.setStyleSheet("color: darkred;")
        layout.addWidget(self.resources_label)

        # Controls
        controls = QHBoxLayout()
        self.toggle_button = QPushButton("Pause detection")
        self.toggle_button.setEnabled(False)
        self.toggle_button.clicked.connect(self._toggle_detection)
        controls.addWidget(self.toggle_button)
        layout.addLayout(controls)

        central.setLayout(layout)
        self.setCentralWidget(central)

    def _on_consent_given(self):
        self.consent_given = True
        self.consent_label.setText("Detection active. You can pause anytime below.")
        self.consent_button.setEnabled(False)
        self.toggle_button.setEnabled(True)
        self.preview_timer.start(UI_REFRESH_INTERVAL_MS)
        self.inference_timer.start(INFERENCE_INTERVAL_MS)

    def _toggle_detection(self):
        if self.inference_timer.isActive():
            self.inference_timer.stop()
            self.toggle_button.setText("Resume detection")
            self.cue_label.setText("Detection paused.")
        else:
            self.inference_timer.start(INFERENCE_INTERVAL_MS)
            self.toggle_button.setText("Pause detection")

    def _update_preview(self):
        success, frame = self.camera.read_frame()
        if not success:
            return
        self._latest_frame = frame
        self._render_frame(frame)

    def _render_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.width(), self.video_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )

    def _run_inference_cycle(self):
        if self._latest_frame is None:
            return
        frame = self._latest_frame

        faces = self.face_detector.detect(frame)
        pose_result = self.pose_detector.detect(frame)

        if not faces:
            self.cue_label.setText("Detected cue: no face currently visible.")
            return

        # Use the highest-confidence face if multiple are detected
        best_face = max(faces, key=lambda f: f["confidence"])
        face_crop = self.face_detector.crop_face(frame, best_face["bbox"])
        label, confidence = self.expression_classifier.classify(face_crop)

        head_down = pose_result["head_down"] if pose_result else False
        still = pose_result["still"] if pose_result else False

        self.aggregator.add_reading(label, confidence, head_down, still)
        aggregate = self.aggregator.get_aggregate()
        response = self.response_engine.evaluate(aggregate)

        self.cue_label.setText(
            f"Detected cue: face consistent with '{label}' "
            f"(estimate confidence: {confidence:.0%})"
        )
        self.response_label.setText(response["message"])

        if response["tier"] == 3 and response["resources"]:
            lines = [f"• {r['name']}: {r['contact']}" for r in response["resources"]]
            self.resources_label.setText("\n".join(lines))
        else:
            self.resources_label.setText("")

    def closeEvent(self, event):
        self.preview_timer.stop()
        self.inference_timer.stop()
        self.camera.release()
        self.face_detector.close()
        self.pose_detector.close()
        event.accept()
