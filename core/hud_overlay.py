import random
from collections import deque
import cv2

from .hud_primitives import (
    HUD_CYAN, HUD_CYAN_DIM, FONT_SMALL,
    BRAIN_SCAN_PATH, BODY_SCAN_PATH,
    _load_rgba, _draw_rgba, _panel_bg, _text,
    _reticle, _waveform_plot, _status_bar, _frame_border
)


class HUDOverlay:
    def __init__(self, waveform_len: int = 50, vitals_update_every_n: int = 6, scale: float = 0.75):
        self.scale = scale
        self.frame_index = 0
        self._waveform_len = waveform_len
        self._expr_wave = deque([0.5] * waveform_len, maxlen=waveform_len)
        self._sim_wave = deque([0.4] * waveform_len, maxlen=waveform_len)
        self._vitals_update_every_n = vitals_update_every_n

        self._vitals = {"hr": 80.0, "spo2": 98.0, "temp": 36.8, "rr": 14.0}
        self._vitals_bounds = {
            "hr": (65.0, 95.0), "spo2": (96.0, 100.0),
            "temp": (36.4, 37.2), "rr": (12.0, 18.0),
        }
        self._vitals_step = {"hr": 1.0, "spo2": 0.2, "temp": 0.03, "rr": 0.3}

        # Smoothed tracking center
        self._smooth_center = None
        self._smooth_radius = None

        # Load scans
        self._brain_img = _load_rgba(BRAIN_SCAN_PATH)
        self._body_img = _load_rgba(BODY_SCAN_PATH)
        if self._body_img is not None:
            # The hologram PNG includes a large transparent canvas. Trim it so its
            # visible body can use the diagnostic panel's intended scale.
            body_bounds = cv2.boundingRect(self._body_img[:, :, 3])
            bx, by, bw, bh = body_bounds
            self._body_img = self._body_img[by:by + bh, bx:bx + bw]

    def _update_fake_vitals(self):
        if self.frame_index % self._vitals_update_every_n != 0:
            return
        for key, (lo, hi) in self._vitals_bounds.items():
            step = self._vitals_step[key]
            self._vitals[key] += random.uniform(-step, step)
            self._vitals[key] = max(lo, min(hi, self._vitals[key]))

    def render(self, frame_bgr, face_bbox, expression_label, expression_confidence,
               head_down, still, tier, tier_message, title="DIAGNOSIS"):
        """Draws the Big Hero 6 diagnostic HUD overlay onto frame_bgr."""
        s = self.scale
        overlay_s = s * 1.5
        h, w = frame_bgr.shape[:2]
        self.frame_index += 1
        self._update_fake_vitals()

        margin = max(14, int(22 * s))

        # -------------------------------------------------------------
        # 1. FACE TRACKING SMOOTHING (STAYS LOCKED ON HEAD) -- unchanged
        # -------------------------------------------------------------
        face_locked = face_bbox is not None
        if face_locked:
            fx, fy, fw, fh = face_bbox
            target_center = (fx + fw // 2, fy + fh // 2)
            target_radius = int(max(fw, fh) * 0.82)
        else:
            target_center = (w // 2, h // 2)
            target_radius = int(min(w, h) * 0.22)

        smooth_alpha = 0.28
        if self._smooth_center is None:
            self._smooth_center = target_center
            self._smooth_radius = target_radius
        else:
            self._smooth_center = (
                int(self._smooth_center[0] + (target_center[0] - self._smooth_center[0]) * smooth_alpha),
                int(self._smooth_center[1] + (target_center[1] - self._smooth_center[1]) * smooth_alpha),
            )
            self._smooth_radius = int(
                self._smooth_radius + (target_radius - self._smooth_radius) * smooth_alpha
            )

        # Keep the pre-existing right-column anchors unchanged.
        left_panel_w = min(int(w * 0.28), int(360 * s))
        left_panel_w = max(left_panel_w, int(255 * s))
        status_bar_h = int(38 * s)
        bottom_floor = h - int(38 * overlay_s) - int(14 * s)
        title_h = int(42 * s)
        header_line_y = margin + title_h + int(10 * s)
        panel_y = header_line_y + int(14 * s)
        inner_pad_y = int(14 * s)
        content_top = panel_y + inner_pad_y
        pad = int(10 * s)

        # -------------------------------------------------------------
        # 2. LEFT DIAGNOSTIC HUD -- compact 1920x1080 reference layout
        # -------------------------------------------------------------
        layout_s = min(w / 1920.0, h / 1080.0)
        lx = lambda value: int(value * layout_s)
        ly = lambda value: int(value * layout_s)

        title_x, title_y = lx(70), ly(72)
        _text(frame_bgr, title, (title_x, title_y), 2.55 * layout_s, HUD_CYAN, 3)
        divider_y = ly(95)
        cv2.line(frame_bgr, (lx(70), divider_y), (lx(520), divider_y), HUD_CYAN_DIM, 1, cv2.LINE_AA)
        cv2.line(frame_bgr, (lx(55), divider_y), (lx(65), divider_y), HUD_CYAN_DIM, 1, cv2.LINE_AA)
        cv2.line(frame_bgr, (lx(520), divider_y), (lx(535), ly(82)), HUD_CYAN_DIM, 1, cv2.LINE_AA)

        # Large hologram anchor, vertically aligned with the adjacent diagnostics.
        body_x, body_y = lx(48), ly(104)
        body_w, body_h = lx(338), ly(593)
        _draw_rgba(frame_bgr, self._body_img, body_x, body_y, body_w, body_h, opacity=0.95)

        symp_x, symp_w = lx(410), lx(405)
        symp_title_y = ly(152)
        _text(frame_bgr, "SYMPTOMS", (symp_x, symp_title_y), 2.22 * layout_s, HUD_CYAN, 2)
        symp_underline_y = ly(175)
        cv2.line(frame_bgr, (symp_x, symp_underline_y), (symp_x + symp_w, symp_underline_y),
                 HUD_CYAN_DIM, 1, cv2.LINE_AA)

        symptoms = [
            f"EXPR: {expression_label.upper()}",
            f"CONF: {expression_confidence * 100:.0f}%",
            f"POSTURE: {'HEAD DOWN' if head_down else 'UPRIGHT'}",
            f"MOTION: {'STILL' if still else 'ACTIVE'}",
            "Vocal Fluctuation",
            "Emotional Instability",
        ]
        line_h = ly(39)
        symp_list_top = ly(210)
        for i, text in enumerate(symptoms):
            _text(frame_bgr, text, (symp_x, symp_list_top + i * line_h),
                  1.72 * layout_s, HUD_CYAN_DIM, 2, FONT_SMALL)

        vit_w, vit_h = lx(428), ly(225)
        # Keep vitals attached to the lower-right HUD edge, above the footer.
        vit_x = w - margin - vit_w
        vit_y = bottom_floor - vit_h
        stat_w = lx(105)
        bp_w = vit_w - stat_w

        _panel_bg(frame_bgr, vit_x, vit_y, bp_w, vit_h, alpha=0.45)
        cv2.rectangle(frame_bgr, (vit_x, vit_y), (vit_x + bp_w, vit_y + vit_h), HUD_CYAN_DIM, 1, cv2.LINE_AA)

        vit_pad = lx(15)
        _text(frame_bgr, "BP   mmHg", (vit_x + vit_pad, vit_y + ly(29)), 1.08 * layout_s, HUD_CYAN_DIM, 1, FONT_SMALL)
        _text(frame_bgr, "113/90", (vit_x + vit_pad, vit_y + ly(84)), 1.95 * layout_s, HUD_CYAN, 2)
        _text(frame_bgr, "80", (vit_x + lx(232), vit_y + ly(84)), 1.95 * layout_s, HUD_CYAN, 2)
        _text(frame_bgr, f"PULSE: {self._vitals['hr']:.0f} bpm", (vit_x + vit_pad, vit_y + ly(132)),
              1.20 * layout_s, HUD_CYAN_DIM, 1, FONT_SMALL)
        _text(frame_bgr, f"MOTION: {'STILL' if still else 'ACTIVE'}", (vit_x + vit_pad, vit_y + ly(168)),
              1.20 * layout_s, HUD_CYAN_DIM, 1, FONT_SMALL)

        stat_x = vit_x + bp_w
        stat_labels = [
            ("RR", f"{self._vitals['rr']:.0f}"),
            ("SPO2", f"{self._vitals['spo2']:.0f}"),
            ("TEMP", f"{self._vitals['temp']:.0f}"),
        ]
        stat_cell_h = vit_h // len(stat_labels)
        for i, (label, value) in enumerate(stat_labels):
            cell_y = vit_y + i * stat_cell_h
            cell_h = stat_cell_h
            _panel_bg(frame_bgr, stat_x, cell_y, stat_w, cell_h, alpha=0.45)
            cv2.rectangle(frame_bgr, (stat_x, cell_y), (stat_x + stat_w, cell_y + cell_h), HUD_CYAN_DIM, 1, cv2.LINE_AA)
            _text(frame_bgr, label, (stat_x + lx(9), cell_y + ly(22)), 0.93 * layout_s, HUD_CYAN_DIM, 1, FONT_SMALL)
            _text(frame_bgr, value, (stat_x + lx(9), cell_y + cell_h - ly(10)), 1.38 * layout_s, HUD_CYAN, 2)

        # -------------------------------------------------------------
        # 3. RIGHT COLUMN LAYOUT
        #    BASELINE brain+graph -> PATIENT brain+graph -> telemetry row
        #    Column position/width is fixed up-front from the frame edge.
        # -------------------------------------------------------------
        right_panel_w = min(int(375 * s), w - margin * 2 - int(200 * s))
        right_panel_w = max(right_panel_w, int(285 * s))
        right_x = w - right_panel_w - margin

        brain_w = int(right_panel_w * 0.42)
        brain_h = int(111 * s)
        graph_x = right_x + brain_w + int(12 * s)
        graph_w = right_panel_w - brain_w - int(12 * s)

        # BASELINE row
        _text(frame_bgr, "BASELINE:", (right_x, content_top + int(9 * s)), 0.78 * s, HUD_CYAN_DIM, 1)
        row1_y = content_top + int(21 * s)
        _draw_rgba(frame_bgr, self._brain_img, right_x, row1_y, brain_w, brain_h, opacity=0.8)
        cv2.rectangle(frame_bgr, (right_x, row1_y), (right_x + brain_w, row1_y + brain_h), HUD_CYAN_DIM, 1)

        _panel_bg(frame_bgr, graph_x, row1_y, graph_w, brain_h, alpha=0.3)
        cv2.rectangle(frame_bgr, (graph_x, row1_y), (graph_x + graph_w, row1_y + brain_h), HUD_CYAN_DIM, 1)
        self._expr_wave.append(min(1.0, max(0.0, expression_confidence)))
        _waveform_plot(frame_bgr, graph_x, row1_y, graph_w, brain_h, list(self._expr_wave), HUD_CYAN, s)

        # PATIENT row
        row2_label_y = row1_y + brain_h + int(24 * s)
        _text(frame_bgr, "PATIENT:", (right_x, row2_label_y), 0.78 * s, HUD_CYAN_DIM, 1)
        row2_y = row2_label_y + int(12 * s)
        _draw_rgba(frame_bgr, self._brain_img, right_x, row2_y, brain_w, brain_h, opacity=0.8)
        cv2.rectangle(frame_bgr, (right_x, row2_y), (right_x + brain_w, row2_y + brain_h), HUD_CYAN_DIM, 1)

        _panel_bg(frame_bgr, graph_x, row2_y, graph_w, brain_h, alpha=0.3)
        cv2.rectangle(frame_bgr, (graph_x, row2_y), (graph_x + graph_w, row2_y + brain_h), HUD_CYAN_DIM, 1)
        sim_val = 0.5 + 0.3 * random.uniform(-1, 1)
        self._sim_wave.append(min(1.0, max(0.0, sim_val)))
        _waveform_plot(frame_bgr, graph_x, row2_y, graph_w, brain_h, list(self._sim_wave), HUD_CYAN_DIM, s)

        # Telemetry row (GnRH / LH / FSH / T / E2 / F)
        chem_y = row2_y + brain_h + int(21 * s)
        chem_h = min(int(69 * s), max(bottom_floor - chem_y, int(54 * s)))
        _panel_bg(frame_bgr, right_x, chem_y, right_panel_w, chem_h, alpha=0.4)
        cv2.rectangle(frame_bgr, (right_x, chem_y), (right_x + right_panel_w, chem_y + chem_h), HUD_CYAN_DIM, 1)

        labels = "GnRH   LH   FSH   T   E2   F"
        values = " 79    81   58  170  22  07"
        _text(frame_bgr, labels, (right_x + int(15 * s), chem_y + int(26 * s)), 1.12 * s, HUD_CYAN_DIM, 1, FONT_SMALL)
        _text(frame_bgr, values, (right_x + int(15 * s), chem_y + int(53 * s)), 1.20 * s, HUD_CYAN, 1, FONT_SMALL)

        # -------------------------------------------------------------
        # 4. CENTER RETICLE (drawn after panels so it always reads on top)
        # -------------------------------------------------------------
        _reticle(frame_bgr, self._smooth_center, self._smooth_radius, self.frame_index, face_locked, s)

        # -------------------------------------------------------------
        # 5. OUTER FRAME + BOTTOM STATUS BAR
        # -------------------------------------------------------------
        floor_y = _status_bar(frame_bgr, tier, tier_message, overlay_s)
        _frame_border(frame_bgr, overlay_s, floor_y)

        return frame_bgr
