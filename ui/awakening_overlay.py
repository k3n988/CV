"""
ui/awakening_overlay.py

A transparent QWidget overlay that renders emotional beats ("Awakening", "Intervention", 
"Anger", "Anxiety", "Disgust", "Embarrassment", and "Connection") on top of the live webcam feed.
"""

import random
from pathlib import Path

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QUrl
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QRadialGradient
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# Default Baymax Theme Colors
INK = QColor(5, 6, 8)
FRAME = QColor(237, 235, 228)
SCAN = QColor(207, 239, 255)
SCAN_GLOW = QColor(127, 217, 255)
MUTED = QColor(124, 139, 153)
HUD = QColor(191, 227, 255)

# Emotional Palette
ALERT = QColor(232, 53, 43)           # Sadness / Intervention
HEAL = QColor(0, 191, 255)            # Joy / Healed
ANGER = QColor(255, 42, 42)           # Anger: High-intensity crimson
ANXIETY = QColor(160, 32, 240)        # Anxiety: Electric violet
DISGUST = QColor(57, 255, 20)         # Disgust: Toxic slime green
EMBARRASSMENT = QColor(255, 105, 180) # Embarrassment: Hot flush pink

# Dynamic Text Captions
CUE1_TEXT = "Hello. I am your personal healthcare companion.\nInitialization complete."
CUE2_TEXT = "Scanning environment.\nMultiple subjects detected."
CUE3_TEXT = (
    "Subject A's neurotransmitter levels indicate\n"
    "profound sadness.\n"
    "This requires immediate intervention."
)
CUE4_TEXT = "It is alright to cry. Crying is a natural response to pain.\nI will provide a comforting embrace."
CUE5_HEAL_TEXT = "Serotonin levels rising.\nProximity to loved ones is highly effective."
CUE6_TEXT = "I cannot deactivate until you say you are satisfied with your care."
CUE7_TEXT = "Balalalala. Powering down..."

ANGER_TEXT = "Adrenaline spike detected. Heart rate elevated.\nPlease take deep, rhythmic breaths to stabilize."
ANXIETY_TEXT = "Rapid shallow breathing detected. Cortisol overload.\nFocus on my breathing rhythm."
DISGUST_TEXT = "Aversion response detected. Sensory overload triggers active.\nRecommending environment adjustment."
EMBARRASSMENT_TEXT = "Facial temperature increase detected (Blushing).\nRest assured, awkwardness is a universal human trait."

POWER_OFF_DELAY_MS = 2500
AUDIO_DIR = Path("assets/audio")

AUDIO_FILES = {
    "cue1": "baymax_greeting.mp3",
    "cue2": "baymax_scanning.mp3",
    "cue3": "baymax_diagnose.mp3",
    "cue4": "baymax_intervention.mp3",
    "cue5": "baymax_healing.mp3",
    "cue6": "baymax_deactivate.mp3",
    "cue7": "baymax_powerdown.mp3",
    "cue_anger": "baymax_anger.mp3",
    "cue_anxiety": "baymax_anxiety.mp3",
    "cue_disgust": "baymax_disgust.mp3",
    "cue_embarrassment": "baymax_embarrassment.mp3",
}


class HeartParticle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'size', 'alpha', 'decay')

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2.5, 2.5)
        self.vy = random.uniform(-4.0, -1.5)
        self.size = random.uniform(10.0, 22.0)
        self.alpha = 1.0
        self.decay = random.uniform(0.015, 0.03)

    def update(self) -> bool:
        self.x += self.vx
        self.y += self.vy
        self.alpha -= self.decay
        return self.alpha > 0.0


class AwakeningOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.stage = "idle"
        self.flash_alpha = 0.0
        self.laser_pct = 0.0
        self.laser_dir = 1
        self.concern_alpha = 0.0
        self.blackout_alpha = 0.0
        self.caption_full = ""
        self.caption_shown = ""
        self.status_word = "STANDBY"
        self.subj_word = "—"

        self.particles = []
        self.glove_scale = 0.0
        self.jitter_offset = QPointF(0, 0)

        # Timers setup
        self._flash_timer = QTimer(self, timeout=self._tick_flash)
        self._laser_timer = QTimer(self, timeout=self._tick_laser)
        self._concern_timer = QTimer(self, timeout=self._tick_concern)
        self._type_timer = QTimer(self, timeout=self._tick_type)
        self._particle_timer = QTimer(self, timeout=self._tick_particles)
        self._glove_timer = QTimer(self, timeout=self._step_glove)
        self._blackout_timer = QTimer(self, timeout=self._step_blackout)

        # Audio playback
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(1.0)
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)

    def showEvent(self, event):
        super().showEvent(event)
        self.reset()

    def closeEvent(self, event):
        self._stop_all_timers()
        self.stop_audio()
        super().closeEvent(event)

    def _stop_all_timers(self):
        self._flash_timer.stop()
        self._laser_timer.stop()
        self._concern_timer.stop()
        self._type_timer.stop()
        self._particle_timer.stop()
        self._glove_timer.stop()
        self._blackout_timer.stop()

    # ---- Audio Controls ---------------------------------------------

    def _play_audio(self, cue_key: str):
        self._player.stop()
        filename = AUDIO_FILES.get(cue_key)
        if not filename:
            return

        path = AUDIO_DIR / filename
        if not path.exists():
            print(f"[AwakeningOverlay] Audio file missing: {path}")
            return

        self._player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self._player.play()

    def stop_audio(self):
        if self._player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
            self._player.stop()

    # ---- Public Cues ------------------------------------------------

    def reset(self):
        """Clears HUD overlay back to idle state and resets animation routines."""
        self.stage = "idle"
        self.flash_alpha = 0.0
        self.laser_pct = 0.0
        self.concern_alpha = 0.0
        self.blackout_alpha = 0.0
        self.status_word = "STANDBY"
        self.subj_word = "—"
        self.particles.clear()
        self.glove_scale = 0.0
        self.jitter_offset = QPointF(0, 0)
        self.caption_full = ""
        self.caption_shown = ""

        self._stop_all_timers()
        self.stop_audio()
        self.update()

    def cue1(self):
        """Initial Greeting & Boot Protocol."""
        self.reset()
        self.stage = "boot"
        self.status_word = "INITIALIZING"
        self.flash_alpha = 1.0
        self._flash_timer.start(16)
        self._set_caption(CUE1_TEXT)
        self._play_audio("cue1")
        QTimer.singleShot(900, lambda: self._set_status("ONLINE"))

    def cue2(self):
        self.reset()
        self.stage = "scan"
        self.status_word = "SCANNING"
        self.subj_word = "3 DETECTED"
        self.laser_dir = 1
        self._laser_timer.start(16)
        self._set_caption(CUE2_TEXT)
        self._play_audio("cue2")

    def cue3(self):
        self.reset()
        self.stage = "diagnosis"
        self.status_word = "ANALYSIS COMPLETE"
        self.subj_word = "SUBJECT A LOCKED"
        self._concern_timer.start(16)
        self._set_caption(CUE3_TEXT)
        self._play_audio("cue3")

    def cue4(self):
        self.reset()
        self.stage = "grief"
        self.concern_alpha = 1.0
        self.status_word = "EMOTIONAL INTERVENTION"
        self.subj_word = "SUBJECT A (GRIEF)"
        self._set_caption(CUE4_TEXT)
        self._play_audio("cue4")

    def cue5(self):
        self.reset()
        self.stage = "hug"
        self.status_word = "EMBRACE DETECTED"
        self.subj_word = "SUBJECTS CONNECTED"
        self._set_caption(CUE5_HEAL_TEXT)
        self._play_audio("cue5")

        cx, cy = self.width() * 0.5, self.height() * 0.45
        self.particles = [HeartParticle(cx, cy) for _ in range(45)]
        self._particle_timer.start(16)

    def cue6(self):
        self.reset()
        self.stage = "prompt"
        self.status_word = "AWAITING RESPONSE"
        self.subj_word = "PATIENT SATISFACTION"
        self._set_caption(CUE6_TEXT)
        self._play_audio("cue6")

    def cue7(self):
        self.reset()
        self.stage = "anger"
        self.status_word = "WARNING: AGITATION"
        self.subj_word = "SUBJECT A (ANGER)"
        self.concern_alpha = 1.0
        self._set_caption(ANGER_TEXT)
        self._play_audio("cue_anger")

    def cue8(self):
        self.reset()
        self.stage = "anxiety"
        self.status_word = "PANIC DETECTED"
        self.subj_word = "SUBJECT A (ANXIETY)"
        self.concern_alpha = 1.0
        self._set_caption(ANXIETY_TEXT)
        self._play_audio("cue_anxiety")

    def cue9(self):
        self.reset()
        self.stage = "disgust"
        self.status_word = "AVERSION DETECTED"
        self.subj_word = "SUBJECT A (DISGUST)"
        self.concern_alpha = 1.0
        self._set_caption(DISGUST_TEXT)
        self._play_audio("cue_disgust")

    def cue0(self):
        self.reset()
        self.stage = "embarrassment"
        self.status_word = "SOCIAL DISCOMFORT"
        self.subj_word = "SUBJECT A (FLUSHED)"
        self.concern_alpha = 1.0
        self._set_caption(EMBARRASSMENT_TEXT)
        self._play_audio("cue_embarrassment")

    def cue_off(self):
        self.reset()
        self.stage = "power_off"
        self.status_word = "DEACTIVATING"
        self.subj_word = "CARE SATISFIED"
        self._set_caption(CUE7_TEXT)
        self._play_audio("cue7")
        QTimer.singleShot(1000, self._trigger_fist_bump)

    # ---- Helpers & Ticks --------------------------------------------

    def _trigger_fist_bump(self):
        if self.stage != "power_off":
            return
        self.glove_scale = 0.05
        self._glove_timer.start(16)

    def _step_glove(self):
        self.glove_scale += 0.06
        if self.glove_scale >= 1.0:
            self.glove_scale = 1.0
            self._glove_timer.stop()
            QTimer.singleShot(POWER_OFF_DELAY_MS, self._trigger_blackout)
        self.update()

    def _trigger_blackout(self):
        self.stage = "blackout"
        self.blackout_alpha = 0.0
        self._blackout_timer.start(16)

    def _step_blackout(self):
        self.blackout_alpha += 0.04
        if self.blackout_alpha >= 1.0:
            self.blackout_alpha = 1.0
            self._blackout_timer.stop()
        self.update()

    def _set_status(self, word: str):
        self.status_word = word
        self.update()

    def _set_caption(self, text: str):
        self.caption_full = text
        self.caption_shown = ""
        self._type_timer.start(22)

    def _tick_type(self):
        n = len(self.caption_shown)
        if n >= len(self.caption_full):
            self._type_timer.stop()
            return
        self.caption_shown = self.caption_full[: n + 1]
        self.update()

    def _tick_flash(self):
        self.flash_alpha -= 0.12
        if self.flash_alpha <= 0.0:
            self.flash_alpha = 0.0
            self._flash_timer.stop()
        self.update()

    def _tick_laser(self):
        step = 0.012
        self.laser_pct += step * self.laser_dir
        if self.laser_pct >= 1.0:
            self.laser_pct = 1.0
            self.laser_dir = -1
        elif self.laser_pct <= 0.0:
            self.laser_pct = 0.0
            self.laser_dir = 1
        self.update()

    def _tick_concern(self):
        self.concern_alpha += 0.04
        if self.concern_alpha >= 1.0:
            self.concern_alpha = 1.0
            self._concern_timer.stop()
        self.update()

    def _tick_particles(self):
        self.particles = [p for p in self.particles if p.update()]
        if not self.particles:
            self._particle_timer.stop()
        self.update()

    # ---- Paint Engine -----------------------------------------------

    def paintEvent(self, event):
        if self.stage == "idle" and self.flash_alpha <= 0:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        if self.stage == "anxiety":
            self.jitter_offset.setX(random.uniform(-3, 3))
            self.jitter_offset.setY(random.uniform(-3, 3))
            p.translate(self.jitter_offset)

        if self.stage != "blackout":
            self._draw_brackets(p, w, h)
            self._draw_hud_readout(p)

        if self.stage == "scan":
            self._draw_laser(p, w, h)

        if self.stage in ("diagnosis", "grief", "anger", "anxiety", "disgust", "embarrassment"):
            self._draw_concern_wash(p, w, h)
            self._draw_vitals(p, w, h)

        if self.stage in ("grief", "hug", "anger", "anxiety", "disgust", "embarrassment"):
            self._draw_subject_tracking(p, w, h)

        if self.stage == "hug" and self.particles:
            self._draw_heart_particles(p)

        if self.stage == "power_off" and self.glove_scale > 0:
            self._draw_baymax_glove(p, w, h)

        self._draw_caption(p, w, h)

        if self.flash_alpha > 0:
            p.fillRect(self.rect(), QColor(255, 255, 255, int(255 * self.flash_alpha)))

        if self.blackout_alpha > 0:
            p.fillRect(self.rect(), QColor(0, 0, 0, int(255 * self.blackout_alpha)))

        p.end()

    def _draw_text_with_shadow(self, p: QPainter, x: int, y: int, text: str, color: QColor):
        p.setPen(QPen(INK))
        p.drawText(x + 1, y + 1, text)
        p.setPen(QPen(color))
        p.drawText(x, y, text)

    def _draw_brackets(self, p: QPainter, w: int, h: int):
        p.setPen(QPen(FRAME, 2))
        p.setOpacity(0.55)
        size = 46
        margin = 28
        p.drawLine(margin, margin, margin + size, margin)
        p.drawLine(margin, margin, margin, margin + size)
        p.drawLine(w - margin, margin, w - margin - size, margin)
        p.drawLine(w - margin, margin, w - margin, margin + size)
        p.drawLine(margin, h - margin, margin + size, h - margin)
        p.drawLine(margin, h - margin, margin, h - margin - size)
        p.drawLine(w - margin, h - margin, w - margin - size, h - margin)
        p.drawLine(w - margin, h - margin, w - margin, h - margin - size)
        p.setOpacity(1.0)

    def _draw_hud_readout(self, p: QPainter):
        p.setOpacity(0.85)
        p.setFont(QFont("Consolas", 10))
        lines = [
            f"SYS.STATUS: {self.status_word}",
            f"TARGET.REF: {self.subj_word}"
        ]
        x, y = 52, 44
        for line in lines:
            self._draw_text_with_shadow(p, x, y, line, HUD)
            y += 18
        p.setOpacity(1.0)

    def _draw_laser(self, p: QPainter, w: int, h: int):
        y = int(self.laser_pct * h)
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor(SCAN_GLOW.red(), SCAN_GLOW.green(), SCAN_GLOW.blue(), 0))
        grad.setColorAt(0.5, SCAN)
        grad.setColorAt(1.0, QColor(SCAN_GLOW.red(), SCAN_GLOW.green(), SCAN_GLOW.blue(), 0))
        
        pen = QPen(grad, 3)
        p.setPen(pen)
        p.drawLine(0, y, w, y)

        trail_top = max(0, y - 140)
        trail = QLinearGradient(0, trail_top, 0, y)
        trail.setColorAt(0.0, QColor(SCAN_GLOW.red(), SCAN_GLOW.green(), SCAN_GLOW.blue(), 0))
        trail.setColorAt(1.0, QColor(SCAN_GLOW.red(), SCAN_GLOW.green(), SCAN_GLOW.blue(), 30))
        p.fillRect(QRectF(0, trail_top, w, y - trail_top), trail)

    def _draw_concern_wash(self, p: QPainter, w: int, h: int):
        color_map = {
            "anger": ANGER,
            "anxiety": ANXIETY,
            "disgust": DISGUST,
            "embarrassment": EMBARRASSMENT,
        }
        color = color_map.get(self.stage, ALERT)

        grad = QRadialGradient(w * 0.5, h * 0.6, max(w, h) * 0.55)
        alpha_val = int(45 * (self.concern_alpha or 1.0))
        grad.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), alpha_val))
        grad.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
        p.fillRect(self.rect(), grad)

    def _draw_vitals(self, p: QPainter, w: int, h: int):
        p.setOpacity(0.9)
        p.setFont(QFont("Consolas", 10))

        vitals_data = {
            "anger": [
                ("SUBJECT A -- VITALS", HUD),
                ("HEART RATE: 145 BPM", ANGER),
                ("BLOOD PRESSURE: ELEVATED", ANGER),
                ("AFFECT: AGITATION", ANGER),
                ("RECOMMEND: DEEP BREATHING", FRAME),
            ],
            "anxiety": [
                ("SUBJECT A -- VITALS", HUD),
                ("RESPIRATION: IRREGULAR", ANXIETY),
                ("CORTISOL LEVEL: PEAK", ANXIETY),
                ("AFFECT: PANIC/ANXIETY", ANXIETY),
                ("RECOMMEND: GROUNDING", FRAME),
            ],
            "disgust": [
                ("SUBJECT A -- VITALS", HUD),
                ("SENSORY INPUT: AVERSIVE", DISGUST),
                ("GUT MOTILITY: SPASMIC", DISGUST),
                ("AFFECT: DISGUST DETECTED", DISGUST),
                ("RECOMMEND: DISTANCING", FRAME),
            ],
            "embarrassment": [
                ("SUBJECT A -- VITALS", HUD),
                ("FACIAL TEMP: 37.8°C (FLUSH)", EMBARRASSMENT),
                ("VASODILATION: HIGH", EMBARRASSMENT),
                ("AFFECT: SOCIAL DISTRESS", EMBARRASSMENT),
                ("RECOMMEND: REASSURANCE", FRAME),
            ],
        }

        default_lines = [
            ("SUBJECT A -- SCAN", HUD),
            ("NEUROTRANSMITTER: LOW", HUD),
            ("CORTISOL: ELEVATED", HUD),
            ("AFFECT: SADNESS DETECTED", ALERT),
            ("ACTION: INTERVENTION REQ.", ALERT),
        ]

        lines = vitals_data.get(self.stage, default_lines)
        x_right = w - 52
        y = h * 0.42
        fm = p.fontMetrics()

        for text, color in lines:
            tw = fm.horizontalAdvance(text)
            tx = int(x_right - tw)
            ty = int(y)
            self._draw_text_with_shadow(p, tx, ty, text, color)
            y += 22
        p.setOpacity(1.0)

    def _draw_subject_tracking(self, p: QPainter, w: int, h: int):
        cx, cy = w * 0.5, h * 0.4
        box_w, box_h = 170, 190
        box_rect = QRectF(cx - box_w / 2, cy - box_h / 2, box_w, box_h)

        badge_map = {
            "hug": (HEAL, "STATUS: HEALED"),
            "anger": (ANGER, "STATUS: ANGER"),
            "anxiety": (ANXIETY, "STATUS: ANXIETY"),
            "disgust": (DISGUST, "STATUS: DISGUST"),
            "embarrassment": (EMBARRASSMENT, "STATUS: FLUSHED"),
        }
        color, label_text = badge_map.get(self.stage, (ALERT, "STATUS: GRIEF"))

        p.setPen(QPen(color, 3 if self.stage == "hug" else 2))
        p.drawRoundedRect(box_rect, 8, 8)

        p.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(label_text)
        badge_rect = QRectF(box_rect.x(), box_rect.y() - 24, tw + 16, 22)
        p.fillRect(badge_rect, color)
        p.setPen(QPen(INK if color == DISGUST else QColor(255, 255, 255)))
        p.drawText(badge_rect, int(Qt.AlignmentFlag.AlignCenter), label_text)

    def _draw_heart_particles(self, p: QPainter):
        p.setFont(QFont("Segoe UI Symbol", 16))
        for pt in self.particles:
            p.setOpacity(max(0.0, min(1.0, pt.alpha)))
            p.setPen(QPen(HEAL))
            p.drawText(int(pt.x), int(pt.y), "♥")
        p.setOpacity(1.0)

    def _draw_baymax_glove(self, p: QPainter, w: int, h: int):
        p.save()
        cx, cy = w * 0.5, h * 0.5
        radius = 180 * self.glove_scale

        grad = QRadialGradient(cx, cy, radius * 1.3)
        grad.setColorAt(0.0, QColor(255, 255, 255, 220))
        grad.setColorAt(0.6, QColor(191, 227, 255, 160))
        grad.setColorAt(1.0, QColor(0, 191, 255, 0))
        p.setBrush(grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), radius * 1.3, radius * 1.3)

        p.setBrush(QColor(250, 252, 255))
        p.setPen(QPen(FRAME, 4))
        p.drawEllipse(QPointF(cx, cy), radius, radius * 0.85)
        p.restore()

    def _draw_caption(self, p: QPainter, w: int, h: int):
        if not self.caption_shown or self.stage == "blackout":
            return
        p.setOpacity(1.0)
        p.setFont(QFont("Segoe UI", 16, QFont.Weight.Medium))

        rect = QRectF(60, h - 160, w - 120, 100)
        flags = int(Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap)

        p.setPen(QPen(QColor(0, 0, 0, 200)))
        p.drawText(rect.translated(1.5, 1.5), flags, self.caption_shown)

        p.setPen(QPen(QColor(255, 255, 255)))
        p.drawText(rect, flags, self.caption_shown)