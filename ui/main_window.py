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
from ui.awakening_overlay import AwakeningOverlay, StartupAwakeningOverlay

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
        self.hand_detector = HandDetector(num_hands=2)
        self.face_detector = FaceDetector()
        self.expression_classifier = ExpressionClassifier()
        self.signal_aggregator = SignalAggregator()
        self.response_engine = ResponseEngine()
        self.hud = HUDOverlay()

        self._build_ui()

        # Instantiate transparent overlay attached on top of video_label
        self.awakening = AwakeningOverlay(self.video_label)
        self.awakening.setGeometry(self.video_label.rect())
        self.awakening.raise_()

        # Created after fullscreen is established so it covers the real display.
        self.startup_overlay = None

        # Synchronize overlay geometry when video label resizes
        self.video_label.resizeEvent = self._on_video_label_resize

        # Single loop timer for detection & rendering
        self.loop_timer = QTimer()
        self.loop_timer.timeout.connect(self._run_loop)

        self._detection_enabled = True
        self._hud_enabled = True
        self.toggle_button.setEnabled(True)
        self.hud_toggle_button.setEnabled(True)
        self.loop_timer.start(LOOP_INTERVAL_MS)

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label, stretch=1)

        controls = QHBoxLayout()
        self.toggle_button = QPushButton("Pause detection")
        self.toggle_button.clicked.connect(self._toggle_detection)
        controls.addWidget(self.toggle_button)

        self.hud_toggle_button = QPushButton("Hide diagnostic HUD")
        self.hud_toggle_button.clicked.connect(self._toggle_hud)
        controls.addWidget(self.hud_toggle_button)

        layout.addLayout(controls)

        # Shortcut guide for cue performance testing
        self.guide_label = QLabel(
            "Hotkeys: [Space] Sequence | [1-6] Story Cues | [7] Anger | [8] Anxiety | [9] Disgust | [0] Embarrass | [.] Power Off | [Esc] Reset"
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

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._show_startup_overlay)

    def _show_startup_overlay(self):
        if self.startup_overlay is not None:
            return
        self.startup_overlay = StartupAwakeningOverlay(self)
        self.startup_overlay.setGeometry(self.rect())
        self.startup_overlay.show()
        self.startup_overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.startup_overlay is not None and self.startup_overlay.isVisible():
            self.startup_overlay.setGeometry(self.rect())
            self.startup_overlay.raise_()

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

            person_landmarks = pose_result.get("landmarks") if pose_result else None
            if person_landmarks:
                display_frame = skeleton_drawer.draw_pose_skeleton(display_frame, person_landmarks)
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
        bytes_per_line = ch * w

        # Clone image buffer safely to prevent memory access glitches in PyQt6
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pixmap.scaled(
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
        key = event.key()

        # Space or Enter: Play sequence or fallback to cue1
        if key in (Qt.Key.Key_Space, Qt.Key.Key_Return):
            if hasattr(self.awakening, 'play_sequence'):
                self.awakening.play_sequence()
            elif hasattr(self.awakening, 'cue1'):
                self.awakening.cue1()

        # Helper method execution for key mappings
        cue_map = {
            Qt.Key.Key_1: 'cue1',
            Qt.Key.Key_2: 'cue2',
            Qt.Key.Key_3: 'cue3',
            Qt.Key.Key_4: 'cue4',
            Qt.Key.Key_5: 'cue5',
            Qt.Key.Key_6: 'cue6',
            Qt.Key.Key_7: 'cue7',
            Qt.Key.Key_8: 'cue8',
            Qt.Key.Key_9: 'cue9',
            Qt.Key.Key_0: 'cue0',
            Qt.Key.Key_Period: 'cue_off',
            Qt.Key.Key_Escape: 'reset',
        }

        if key in cue_map:
            method_name = cue_map[key]
            if hasattr(self.awakening, method_name):
                getattr(self.awakening, method_name)()
        else:
            super().keyPressEvent(event)
