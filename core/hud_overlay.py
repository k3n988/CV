"""
hud_overlay.py

Draws a sci-fi "diagnostic HUD" overlay on top of the webcam feed --
a circular reticle locked onto the detected face, a left-side signal
panel, right-side waveform panels, a bottom vitals grid, and a status
bar driven by the app's real ResponseEngine tier.

All panel/text/reticle sizing is driven by a single `scale` factor
(see HUDOverlay.__init__) so the whole HUD can be made larger or
smaller without touching the drawing code -- current default targets
a compact-but-readable look on a 1280x720 widescreen feed.

Data honesty note: the "signal panel" and "assessment" text are driven
by real values from ExpressionClassifier / PoseDetector / ResponseEngine.
The vitals grid (pulse, SpO2, temp, RR) and the "ACTIVITY (SIM)" waveform
are cosmetic only -- a webcam cannot measure blood oxygen or temperature.
They're clearly labeled "SIMULATED" in the panel itself so the HUD never
implies it's reading real biometrics it isn't.
"""

import math
import random
from collections import deque

import cv2

# ---- Palette (BGR) ---------------------------------------------------
HUD_CYAN = (255, 220, 60)
HUD_CYAN_DIM = (170, 140, 40)
HUD_AMBER = (40, 170, 255)
HUD_RED = (60, 60, 255)
PANEL_BG = (35, 25, 15)

TIER_COLORS = {1: HUD_CYAN, 2: HUD_AMBER, 3: HUD_RED}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

# Minimum font scale/thickness floors so text never becomes unreadable
# even if a very small `scale` is passed in.
MIN_FONT_SCALE = 0.42


def _panel_bg(frame, x, y, w, h, alpha=0.45):
    """Draws a translucent dark rectangle behind panel content."""
    x2, y2 = x + w, y + h
    x, y = max(x, 0), max(y, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 <= x or y2 <= y:
        return
    roi = frame[y:y2, x:x2]
    overlay = roi.copy()
    overlay[:] = PANEL_BG
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _text(frame, text, org, scale=0.5, color=HUD_CYAN, thickness=1, font=FONT):
    scale = max(scale, MIN_FONT_SCALE)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _corner_brackets(frame, s, length=26, thickness=2, margin=14, color=HUD_CYAN_DIM):
    length, margin = int(length * s), int(margin * s)
    h, w = frame.shape[:2]
    corners = [
        ((margin, margin), (1, 0), (0, 1)),
        ((w - margin, margin), (-1, 0), (0, 1)),
        ((margin, h - margin), (1, 0), (0, -1)),
        ((w - margin, h - margin), (-1, 0), (0, -1)),
    ]
    for (cx, cy), dx, dy in corners:
        p1 = (cx, cy)
        p2 = (cx + dx[0] * length, cy + dx[1] * length)
        p3 = (cx + dy[0] * length, cy + dy[1] * length)
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.line(frame, p1, p3, color, thickness, cv2.LINE_AA)


def _reticle(frame, center, radius, frame_index, locked, s):
    cx, cy = center
    color = HUD_CYAN if locked else HUD_CYAN_DIM
    tick_len = max(4, int(8 * s))
    dot_r = max(2, int(4 * s))

    cv2.circle(frame, (cx, cy), radius, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), int(radius * 0.72), color, 1, cv2.LINE_AA)

    # Tick marks every 30 degrees
    for deg in range(0, 360, 30):
        rad = math.radians(deg)
        x1 = int(cx + math.cos(rad) * radius)
        y1 = int(cy + math.sin(rad) * radius)
        x2 = int(cx + math.cos(rad) * (radius + tick_len))
        y2 = int(cy + math.sin(rad) * (radius + tick_len))
        cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Small filled dots at top and bottom, like a targeting reticle
    cv2.circle(frame, (cx, cy - radius), dot_r, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy + radius), dot_r, color, -1, cv2.LINE_AA)

    # Rotating scan arc for a "live" feel
    if locked:
        start = (frame_index * 4) % 360
        arc_offset = max(10, int(14 * s))
        cv2.ellipse(
            frame, (cx, cy), (radius + arc_offset, radius + arc_offset), 0,
            start, start + 50, HUD_CYAN, 2, cv2.LINE_AA,
        )
    else:
        _text(frame, "SEARCHING...", (cx - int(55 * s), cy + radius + int(26 * s)),
              0.5 * s, HUD_CYAN_DIM)


def _signal_panel(frame, x, y, w, h, face_locked, expression_label,
                   expression_confidence, head_down, still, s):
    _panel_bg(frame, x, y, w, h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), HUD_CYAN_DIM, 1, cv2.LINE_AA)
    pad = int(10 * s)
    _text(frame, "SIGNALS", (x + pad, y + int(22 * s)), 0.55 * s, HUD_CYAN, 1)
    cv2.line(frame, (x + pad, y + int(30 * s)), (x + w - pad, y + int(30 * s)),
              HUD_CYAN_DIM, 1)

    lines = [
        f"FACE     : {'LOCKED' if face_locked else 'SEARCHING'}",
        f"EXPR     : {expression_label.upper()} ({expression_confidence * 100:.0f}%)",
        f"POSTURE  : {'HEAD DOWN' if head_down else 'UPRIGHT'}",
        f"MOTION   : {'STILL' if still else 'ACTIVE'}",
    ]
    line_h = int(18 * s)
    for i, line in enumerate(lines):
        _text(frame, line, (x + pad, y + int(50 * s) + i * line_h), 1.0 * s,
              HUD_CYAN, 1, FONT_SMALL)


def _waveform_panel(frame, x, y, w, h, values, color, label, s):
    _panel_bg(frame, x, y, w, h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), HUD_CYAN_DIM, 1, cv2.LINE_AA)
    _text(frame, label, (x + int(8 * s), y + int(16 * s)), 1.0 * s,
          HUD_CYAN_DIM, 1, FONT_SMALL)

    plot_top = y + int(22 * s)
    plot_h = h - int(28 * s)
    if len(values) >= 2 and plot_h > 2:
        pts = []
        for i, v in enumerate(values):
            px = x + int(i / (len(values) - 1) * (w - int(16 * s))) + int(8 * s)
            py = plot_top + plot_h - int(v * plot_h)
            pts.append((px, py))
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)


def _vitals_grid(frame, x, y, w, h, vitals, s):
    _panel_bg(frame, x, y, w, h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), HUD_CYAN_DIM, 1, cv2.LINE_AA)

    cells = [
        ("PULSE", f"{vitals['hr']:.0f}", "bpm"),
        ("SPO2", f"{vitals['spo2']:.0f}", "%"),
        ("TEMP", f"{vitals['temp']:.1f}", "C"),
        ("RR", f"{vitals['rr']:.0f}", "/min"),
    ]
    cell_w = w // 2
    cell_h = int(32 * s)
    pad = int(8 * s)
    for i, (label, value, unit) in enumerate(cells):
        cx = x + (i % 2) * cell_w
        cy = y + pad + (i // 2) * cell_h
        _text(frame, label, (cx + pad, cy + int(10 * s)), 0.85 * s,
              HUD_CYAN_DIM, 1, FONT_SMALL)
        _text(frame, f"{value} {unit}", (cx + pad, cy + int(26 * s)), 1.1 * s,
              HUD_CYAN, 1, FONT_SMALL)

    _text(frame, "SIMULATED -- NOT A MEDICAL DEVICE", (x + pad, y + h - int(6 * s)),
          0.8 * s, HUD_CYAN_DIM, 1, FONT_SMALL)


def _status_bar(frame, tier, message, s):
    h_frame, w_frame = frame.shape[:2]
    bar_h = int(44 * s)
    y = h_frame - bar_h
    color = TIER_COLORS.get(tier, HUD_CYAN)
    _panel_bg(frame, 0, y, w_frame, bar_h, alpha=0.55)
    cv2.line(frame, (0, y), (w_frame, y), color, 1, cv2.LINE_AA)

    label = f"ASSESSMENT (TIER {tier}): "
    full_text = label + message
    font_scale = max(1.1 * s, MIN_FONT_SCALE)

    # Simple wrap-or-truncate so long messages don't overflow the bar.
    (text_w, _), _ = cv2.getTextSize(full_text, FONT_SMALL, font_scale, 1)
    max_w = w_frame - int(24 * s)
    if text_w > max_w:
        while full_text and cv2.getTextSize(full_text + "...", FONT_SMALL, font_scale, 1)[0][0] > max_w:
            full_text = full_text[:-1]
        full_text += "..."

    _text(frame, full_text, (int(12 * s), y + int(28 * s)), font_scale, color, 1, FONT_SMALL)


class HUDOverlay:
    """
    Stateful HUD renderer. Create one instance and call render() once per
    frame -- it tracks its own animation counter and slow-moving fake
    vitals/waveform history internally so the overlay looks "alive"
    across frames instead of static.

    `scale` controls the overall size of every HUD element (panels,
    text, reticle). 1.0 is the original full size; the default of 0.7
    gives a noticeably more compact HUD that still stays readable at
    typical webcam-window sizes. Lower it further (e.g. 0.55) for an
    even smaller HUD, or raise it (e.g. 1.0+) for a larger one -- text
    has a hard readability floor built in so it won't vanish.
    """

    def __init__(self, waveform_len: int = 60, vitals_update_every_n: int = 6,
                 scale: float = 0.7):
        self.scale = scale
        self.frame_index = 0
        self._waveform_len = waveform_len
        self._expr_wave = deque([0.0] * waveform_len, maxlen=waveform_len)
        self._sim_wave = deque([0.5] * waveform_len, maxlen=waveform_len)
        self._vitals_update_every_n = vitals_update_every_n

        self._vitals = {"hr": 72.0, "spo2": 98.0, "temp": 36.6, "rr": 15.0}
        self._vitals_bounds = {
            "hr": (58.0, 92.0), "spo2": (95.0, 100.0),
            "temp": (36.2, 37.1), "rr": (11.0, 19.0),
        }
        self._vitals_step = {"hr": 1.2, "spo2": 0.3, "temp": 0.04, "rr": 0.4}

    def _update_fake_vitals(self):
        if self.frame_index % self._vitals_update_every_n != 0:
            return
        for key, (lo, hi) in self._vitals_bounds.items():
            step = self._vitals_step[key]
            self._vitals[key] += random.uniform(-step, step)
            self._vitals[key] = max(lo, min(hi, self._vitals[key]))

    def render(self, frame_bgr, face_bbox, expression_label, expression_confidence,
               head_down, still, tier, tier_message, title="STATUS"):
        """
        Draws the full HUD onto frame_bgr in place and returns it.

        face_bbox: (x, y, w, h) in frame_bgr's pixel space, or None.
        expression_label/confidence: from ExpressionClassifier.classify().
        head_down/still: from PoseDetector.detect() (False if unavailable).
        tier/tier_message: from ResponseEngine.evaluate().
        """
        s = self.scale
        h, w = frame_bgr.shape[:2]
        self.frame_index += 1
        self._update_fake_vitals()

        face_locked = face_bbox is not None
        if face_locked:
            fx, fy, fw, fh = face_bbox
            center = (fx + fw // 2, fy + fh // 2)
            radius = int(max(fw, fh) * 0.85)
        else:
            center = (w // 2, h // 2)
            radius = int(min(w, h) * 0.22)

        # Scrolling waveforms: one tied to a real signal, one decorative.
        self._expr_wave.append(min(1.0, max(0.0, expression_confidence)))
        sim_val = 0.5 + 0.4 * random.uniform(-1, 1) * 0.3 + 0.1 * (self.frame_index % 20) / 20
        self._sim_wave.append(min(1.0, max(0.0, sim_val)))

        panel_w = max(250, int(280 * s))
        signal_h = max(120, int(150 * s))
        wave_h = max(80, int(100 * s))
        vitals_h = max(84, int(112 * s))
        margin = max(10, int(18 * s))
        gap = max(4, int(6 * s))

        _corner_brackets(frame_bgr, s)
        _reticle(frame_bgr, center, radius, self.frame_index, face_locked, s)

        title_scale = max(1.0 * s, MIN_FONT_SCALE)
        _text(frame_bgr, title, (margin, margin + int(20 * s)), title_scale, HUD_CYAN, 2)
        cv2.line(frame_bgr, (margin, margin + int(28 * s)),
                  (margin + int(140 * s), margin + int(28 * s)), HUD_CYAN_DIM, 1)

        signal_y = margin + int(36 * s)
        _signal_panel(frame_bgr, margin, signal_y, panel_w, signal_h, face_locked,
                      expression_label, expression_confidence, head_down, still, s)

        right_x = w - panel_w - margin
        _waveform_panel(frame_bgr, right_x, margin, panel_w, wave_h, list(self._expr_wave),
                         HUD_CYAN, "EXPR CONFIDENCE", s)
        _waveform_panel(frame_bgr, right_x, margin + wave_h + gap, panel_w, wave_h,
                         list(self._sim_wave), HUD_CYAN_DIM, "ACTIVITY (SIM)", s)

        bar_h = int(44 * s)
        _vitals_grid(frame_bgr, margin, h - bar_h - vitals_h - gap * 2, panel_w, vitals_h,
                     self._vitals, s)

        _status_bar(frame_bgr, tier, tier_message, s)

        return frame_bgr