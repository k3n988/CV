"""
main_window.py

PyQt6 desktop UI: shows the live webcam feed with a body pose and
hand/finger skeleton overlay, running detection on every rendered
frame (single loop) to minimize perceived lag between real movement
and the on-screen overlay. Optionally also renders a sci-fi
"diagnostic HUD" (core/hud_overlay.py) driven by the same
face/expression/posture signals used by the response engine.
and the on-screen overlay.
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

# Single loop interval -- lower = more responsive, but more CPU load per
# second. 33ms (~30fps) matches typical webcam frame rate; going lower
# than your webcam's native fps won't help since there's no new frame yet.
LOOP_INTERVAL_MS = 33

# Detection runs on a downscaled copy of the frame for speed, then
# coordinates are naturally handled since MediaPipe landmarks are
# normalized (0.0-1.0), not pixel-based -- so downscaling doesn't need
# any coordinate correction.
DETECTION_SCALE = 0.5  # process at half resolution


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EmoSense -- Pose & Hand Skeleton Demo")
        self.setMinimumSize(760, 640)

        # --- Pipeline components ---
        self.camera = CameraCapture()
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector(num_hands=10)  # set to 2 if you need both hands
        self.face_detector = FaceDetector()
        self.expression_classifier = ExpressionClassifier()  # stub mode if no model file yet
        self.signal_aggregator = SignalAggregator()
        self.response_engine = ResponseEngine()
        self.hud = HUDOverlay()


        self.consent_given = False
        self._build_ui()

        # Single timer drives both the video render AND detection --
        # no separate slower "inference cycle" lagging behind the video.
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

        central.setLayout(layout)
        self.setCentralWidget(central)

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
            # Detect on a smaller frame for speed -- landmarks are
            # normalized coordinates, so they map directly onto the
            # full-size display_frame without any rescaling math.
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

            # face_bbox_small is in small_frame's pixel space -- scale it
            # back up to full-resolution display_frame coordinates (same
            # normalization-free approach as the skeleton overlays, just
            # done manually since face bboxes are pixel-based, not
            # normalized like MediaPipe landmarks).
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