"""
main_window.py

PyQt6 desktop UI: shows the live webcam feed with a body pose and
hand/finger skeleton overlay, running detection on every rendered
frame (single loop) to minimize perceived lag between real movement
and the on-screen overlay. Optionally renders a sci-fi "diagnostic HUD"
driven by the response engine signals and the transparent Awakening overlay.
"""

import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

from core.camera_capture import CameraCapture
from core.pose_detector import PoseDetector
from core.hand_detector import HandDetector
from core.face_detector import FaceDetector
from core.expression_classifier import ExpressionClassifier
from core.signal_aggregator import SignalAggregator
from core.response_engine import ResponseEngine
from core.hud_overlay import HUDOverlay
from core import skeleton_drawer
from ui.awakening_overlay import AwakeningOverlay

# Single loop interval -- 33ms (~30fps) matches typical webcam frame rate
LOOP_INTERVAL_MS = 33

# Detection runs on a downscaled copy of the frame for speed
DETECTION_SCALE = 0.5  # process at half resolution


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EmoSense -- Pose & Hand Skeleton Demo")
        self.setMinimumSize(760, 640)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # --- Pipeline components ---
        self.camera = CameraCapture()
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector(num_hands=10)
        self.face_detector = FaceDetector()
        self.expression_classifier = ExpressionClassifier()
        self.signal_aggregator = SignalAggregator()
        self.response_engine = ResponseEngine()
        self.hud = HUDOverlay()

        self.consent_given = False
        self._build_ui()

        # Instantiate transparent overlay attached on top of video_label
        self.awakening = AwakeningOverlay(self.video_label)
        self.awakening.setGeometry(self.video_label.rect())
        self.awakening.raise_()

        # Synchronize overlay geometry when video label resizes
        self.video_label.resizeEvent = self._on_video_label_resize

        # Single loop timer for detection & rendering
        self.loop_timer = QTimer()
        self.loop_timer.timeout.connect(self._run_loop)

        self._detection_enabled = True
        self._hud_enabled = True

        self.showMaximized()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.consent_label = QLabel(
            "This demo uses your webcam to show a live body pose and hand "
            "skeleton overlay. No video is stored or transmitted; everything "
            "runs on this device, this session only."
        )
        self.consent_label.setWordWrap(True)
        layout.addWidget(self.consent_label)

        self.consent_button = QPushButton("I understand -- start detection")
        self.consent_button.clicked.connect(self._on_consent_given)
        layout.addWidget(self.consent_button)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label, stretch=1)

        controls = QHBoxLayout()
        self.toggle_button = QPushButton("Pause detection")
        self.toggle_button.setEnabled(False)
        self.toggle_button.clicked.connect(self._toggle_detection)
        controls.addWidget(self.toggle_button)

        self.hud_toggle_button = QPushButton("Hide diagnostic HUD")
        self.hud_toggle_button.setEnabled(False)
        self.hud_toggle_button.clicked.connect(self._toggle_hud)
        controls.addWidget(self.hud_toggle_button)

        layout.addLayout(controls)

        # Shortcut guide for cue performance testing
        self.guide_label = QLabel(
            "Performance Hotkeys: [Space] Play Full Sequence | [1] Boot | [2] Scan | [3] Diagnosis | [4] Grief | [5] Heal/Embrace | [6] Fist Bump Climax | [Esc/0] Reset"
        )
        self.guide_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.guide_label.setStyleSheet("color: #7C8B99; font-size: 11px;")
        layout.addWidget(self.guide_label)

        central.setLayout(layout)
        self.setCentralWidget(central)

    def _on_video_label_resize(self, event):
        """Keep the overlay perfectly stretched to the video_label dimensions."""
        if hasattr(self, 'awakening'):
            self.awakening.setGeometry(self.video_label.rect())
            self.awakening.raise_()
        if event:
            super(QLabel, self.video_label).resizeEvent(event)

    def _on_consent_given(self):
        self.consent_given = True
        self.consent_label.hide()
        self.consent_button.hide()
        self.toggle_button.setEnabled(True)
        self.hud_toggle_button.setEnabled(True)
        self.loop_timer.start(LOOP_INTERVAL_MS)

    def _toggle_detection(self):
        self._detection_enabled = not self._detection_enabled
        self.toggle_button.setText(
            "Pause detection" if self._detection_enabled else "Resume detection"
        )

    def _toggle_hud(self):
        self._hud_enabled = not self._hud_enabled
        self.hud_toggle_button.setText(
            "Hide diagnostic HUD" if self._hud_enabled else "Show diagnostic HUD"
        )

    def _run_loop(self):
        success, frame = self.camera.read_frame()
        if not success:
            return

        display_frame = frame.copy()

        if self._detection_enabled:
            small_frame = cv2.resize(
                frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE,
                interpolation=cv2.INTER_LINEAR,
            )

            pose_result = self.pose_detector.detect(small_frame)
            hands_result = self.hand_detector.detect(small_frame)
            face_bbox_small = self.face_detector.detect(small_frame)

            pose_landmarks = pose_result["landmarks"] if pose_result else None
            display_frame = skeleton_drawer.draw_pose_skeleton(display_frame, pose_landmarks)
            display_frame = skeleton_drawer.draw_hand_skeleton(display_frame, hands_result)

            head_down = pose_result["head_down"] if pose_result else False
            still = pose_result["still"] if pose_result else False

            face_bbox_full = None
            expression_label, expression_confidence = "neutral", 0.0
            if face_bbox_small is not None:
                inv_scale = 1.0 / DETECTION_SCALE
                fx, fy, fw, fh = face_bbox_small
                fx, fy = int(fx * inv_scale), int(fy * inv_scale)
                fw, fh = int(fw * inv_scale), int(fh * inv_scale)
                h, w = display_frame.shape[:2]
                fx, fy = max(0, fx), max(0, fy)
                fw, fh = min(fw, w - fx), min(fh, h - fy)
                if fw > 0 and fh > 0:
                    face_bbox_full = (fx, fy, fw, fh)
                    face_crop = display_frame[fy:fy + fh, fx:fx + fw]
                    expression_label, expression_confidence = (
                        self.expression_classifier.classify(face_crop)
                    )

            self.signal_aggregator.add_reading(
                expression_label, expression_confidence, head_down, still
            )
            aggregate = self.signal_aggregator.get_aggregate()
            response = self.response_engine.evaluate(aggregate)

            if self._hud_enabled:
                display_frame = self.hud.render(
                    display_frame,
                    face_bbox_full,
                    expression_label,
                    expression_confidence,
                    head_down,
                    still,
                    response["tier"],
                    response["message"],
                )

        self._render_frame(display_frame)

    def _render_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def closeEvent(self, event):
        self.loop_timer.stop()
        self.camera.release()
        self.pose_detector.close()
        self.hand_detector.close()
        self.face_detector.close()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return):
            if hasattr(self.awakening, 'play_sequence'):
                self.awakening.play_sequence()
            else:
                self.awakening.cue1()
        elif event.key() == Qt.Key.Key_1:
            self.awakening.cue1()
        elif event.key() == Qt.Key.Key_2:
            self.awakening.cue2()
        elif event.key() == Qt.Key.Key_3:
            self.awakening.cue3()
        elif event.key() == Qt.Key.Key_4:
            self.awakening.cue4()
        elif event.key() == Qt.Key.Key_5:
            self.awakening.cue5()
        elif event.key() == Qt.Key.Key_6:
            self.awakening.cue6()
        elif event.key() == Qt.Key.Key_7:
                    self.awakening.cue7()    
        elif event.key() in (Qt.Key.Key_0, Qt.Key.Key_Escape):
            self.awakening.reset()
        else:
            super().keyPressEvent(event)