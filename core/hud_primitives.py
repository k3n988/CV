import math
import os
import cv2

# ---- Palette (BGR Sci-Fi Cyan/Blue) -----------------------------------
HUD_CYAN = (255, 235, 100)
HUD_CYAN_DIM = (180, 150, 40)
HUD_BLUE_GLOW = (255, 200, 50)
HUD_AMBER = (40, 170, 255)
HUD_RED = (60, 60, 255)
PANEL_BG = (25, 18, 10)

TIER_COLORS = {1: HUD_CYAN, 2: HUD_AMBER, 3: HUD_RED}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

MIN_FONT_SCALE = 0.38

# ---- Decorative Asset Paths -------------------------------------------
ASSETS_DIR = "assets"
BRAIN_SCAN_PATH = os.path.join(ASSETS_DIR, "brain_scan.png")
BODY_SCAN_PATH = os.path.join(ASSETS_DIR, "body_scan.png")


def _load_rgba(path):
    """Loads an image preserving alpha transparency if present."""
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _draw_rgba(frame, rgba_img, x, y, target_w, target_h, opacity=0.85):
    """Alpha-blends an RGBA image onto frame_bgr at (x, y), centered inside
    the (target_w, target_h) box while preserving the source aspect ratio
    so hologram assets never look squashed or oddly cropped."""
    if rgba_img is None or target_w <= 0 or target_h <= 0:
        return
    h_frame, w_frame = frame.shape[:2]

    src_h, src_w = rgba_img.shape[:2]
    src_aspect = src_w / max(src_h, 1)
    box_aspect = target_w / max(target_h, 1)

    if src_aspect > box_aspect:
        draw_w = target_w
        draw_h = max(1, int(target_w / src_aspect))
    else:
        draw_h = target_h
        draw_w = max(1, int(target_h * src_aspect))

    off_x = x + (target_w - draw_w) // 2
    off_y = y + (target_h - draw_h) // 2

    resized = cv2.resize(rgba_img, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
    b, g, r, a = cv2.split(resized)
    alpha = (a.astype(float) / 255.0) * opacity

    x, y = off_x, off_y
    x2, y2 = x + draw_w, y + draw_h
    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = draw_w, draw_h

    if x < 0:
        src_x1 = -x
        x = 0
    if y < 0:
        src_y1 = -y
        y = 0
    if x2 > w_frame:
        src_x2 -= (x2 - w_frame)
        x2 = w_frame
    if y2 > h_frame:
        src_y2 -= (y2 - h_frame)
        y2 = h_frame

    if x2 <= x or y2 <= y or src_x2 <= src_x1 or src_y2 <= src_y1:
        return

    roi = frame[y:y2, x:x2].astype(float)
    fg = cv2.merge([b, g, r])[src_y1:src_y2, src_x1:src_x2].astype(float)
    a_crop = alpha[src_y1:src_y2, src_x1:src_x2][..., None]

    blended = fg * a_crop + roi * (1 - a_crop)
    frame[y:y2, x:x2] = blended.astype("uint8")


def _panel_bg(frame, x, y, w, h, alpha=0.35):
    """Draws a translucent dark background behind HUD elements."""
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


def _text_width(text, scale, thickness=1, font=FONT):
    scale = max(scale, MIN_FONT_SCALE)
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    return tw


def _reticle(frame, center, radius, frame_index, locked, s):
    """Face Tracking Reticle: Smoothly follows user face/head position.
    (Unchanged tracking/visual behavior -- position math is untouched.)"""
    cx, cy = center
    color = HUD_CYAN if locked else HUD_CYAN_DIM
    tick_len = max(4, int(10 * s))
    dot_r = max(2, int(5 * s))

    # Outer main circle and inner dashed target ring
    cv2.circle(frame, (cx, cy), radius, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), int(radius * 0.75), color, 1, cv2.LINE_AA)

    # Crosshair tick marks around reticle ring
    for deg in range(0, 360, 45):
        rad = math.radians(deg)
        x1 = int(cx + math.cos(rad) * (radius - tick_len // 2))
        y1 = int(cy + math.sin(rad) * (radius - tick_len // 2))
        x2 = int(cx + math.cos(rad) * (radius + tick_len))
        y2 = int(cy + math.sin(rad) * (radius + tick_len))
        cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Movie-accurate top and bottom notch dots
    cv2.circle(frame, (cx, cy - radius), dot_r, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy + radius), dot_r, color, -1, cv2.LINE_AA)

    # Active scanning arc animation
    if locked:
        start = (frame_index * 5) % 360
        arc_offset = max(8, int(12 * s))
        cv2.ellipse(
            frame, (cx, cy), (radius + arc_offset, radius + arc_offset), 0,
            start, start + 60, HUD_CYAN, 2, cv2.LINE_AA,
        )
    else:
        _text(frame, "TARGET LOCKING...", (cx - int(55 * s), cy + radius + int(24 * s)),
              0.45 * s, HUD_CYAN_DIM)


def _waveform_plot(frame, x, y, w, h, values, color, s):
    """Draws a sci-fi line graph waveform."""
    if len(values) < 2 or h <= 2:
        return
    pts = []
    plot_top = y + 4
    plot_h = h - 8
    for i, v in enumerate(values):
        px = x + int(i / (len(values) - 1) * w)
        py = plot_top + plot_h - int(v * plot_h)
        pts.append((px, py))
    for p1, p2 in zip(pts, pts[1:]):
        cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)


def _status_bar(frame, tier, message, s):
    """Bottom diagnostic banner."""
    h_frame, w_frame = frame.shape[:2]
    bar_h = int(38 * s)
    y = h_frame - bar_h
    color = TIER_COLORS.get(tier, HUD_CYAN)

    _panel_bg(frame, 0, y, w_frame, bar_h, alpha=0.6)
    cv2.line(frame, (0, y), (w_frame, y), color, 1, cv2.LINE_AA)

    full_text = f"BAYMAX DIAGNOSTIC [TIER {tier}]: {message}"
    font_scale = max(0.9 * s, MIN_FONT_SCALE)

    (text_w, _), _ = cv2.getTextSize(full_text, FONT_SMALL, font_scale, 1)
    max_w = w_frame - int(24 * s)
    if text_w > max_w:
        while full_text and cv2.getTextSize(full_text + "...", FONT_SMALL, font_scale, 1)[0][0] > max_w:
            full_text = full_text[:-1]
        full_text += "..."

    _text(frame, full_text, (int(15 * s), y + int(24 * s)), font_scale, color, 1, FONT_SMALL)

    return y  # top of status bar, so panels above know where the floor is


def _frame_border(frame, s, floor_y):
    """Thin outer border + corner brackets so the whole feed reads as one
    contained diagnostic screen (matches the bracket/frame look of the
    reference image)."""
    h, w = frame.shape[:2]
    margin = max(6, int(8 * s))
    bracket = max(16, int(28 * s))
    color = HUD_CYAN_DIM
    thickness = 1

    # Full border, stopping above the status bar so it doesn't double-line it
    cv2.rectangle(frame, (margin, margin), (w - margin, floor_y - margin // 2),
                  color, thickness, cv2.LINE_AA)

    corners = [
        (margin, margin, 1, 1),
        (w - margin, margin, -1, 1),
        (margin, floor_y - margin // 2, 1, -1),
        (w - margin, floor_y - margin // 2, -1, -1),
    ]
    for cx, cy, dx, dy in corners:
        cv2.line(frame, (cx, cy), (cx + dx * bracket, cy), HUD_CYAN, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * bracket), HUD_CYAN, 2, cv2.LINE_AA)