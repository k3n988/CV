"""
skeleton_drawer.py

Draws stick-figure style overlays -- connector lines + joint dots --
for a detected face box, body pose, and hand/finger landmarks directly
onto a video frame. This is purely visual (for the presentation demo);
it does not affect the expression/posture signals used by the response
engine.
"""

import cv2

# Simplified BlazePose (33-point) connections -- enough to show a clear
# head/shoulders/arms/torso/legs stick figure without cluttering the
# frame with all 33 points.
POSE_BODY_CONNECTIONS = [
    (11, 12),            # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),            # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]

# Loose head connections (ear - eye - nose - eye - ear) to suggest a
# head shape above the shoulder line, similar to reference pose diagrams.
POSE_HEAD_CONNECTIONS = [
    (7, 2), (2, 0), (0, 5), (5, 8),
]

_POSE_ALL_CONNECTIONS = POSE_BODY_CONNECTIONS + POSE_HEAD_CONNECTIONS
POSE_JOINTS_TO_DRAW = sorted(set(idx for pair in _POSE_ALL_CONNECTIONS for idx in pair))

# Standard MediaPipe Hands connections (21-point hand model)
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),  # palm
    (1, 2), (2, 3), (3, 4),          # thumb
    (5, 6), (6, 7), (7, 8),          # index finger
    (9, 10), (10, 11), (11, 12),     # middle finger
    (13, 14), (14, 15), (15, 16),    # ring finger
    (17, 18), (18, 19), (19, 20),    # pinky finger
]

LINE_COLOR = (255, 255, 255)   # white connector lines (BGR)
JOINT_COLOR = (0, 90, 245)     # orange-red joint dots (BGR)
FACE_BOX_COLOR = (0, 90, 245)
LINE_THICKNESS = 2
JOINT_RADIUS = 5
HAND_JOINT_RADIUS = 4


def _to_pixel(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def draw_pose_skeleton(frame_bgr, landmarks):
    """Draws a simplified body skeleton overlay from BlazePose landmarks."""
    if landmarks is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]

    for a, b in _POSE_ALL_CONNECTIONS:
        pa = _to_pixel(landmarks[a], w, h)
        pb = _to_pixel(landmarks[b], w, h)
        cv2.line(frame_bgr, pa, pb, LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)

    for idx in POSE_JOINTS_TO_DRAW:
        p = _to_pixel(landmarks[idx], w, h)
        cv2.circle(frame_bgr, p, JOINT_RADIUS, JOINT_COLOR, -1, cv2.LINE_AA)

    return frame_bgr


def draw_hand_skeleton(frame_bgr, hands_landmarks):
    """
    Draws hand/finger skeleton overlays. hands_landmarks is a list of
    landmark lists (one per detected hand), each with 21 points.
    """
    if not hands_landmarks:
        return frame_bgr
    h, w = frame_bgr.shape[:2]

    for hand in hands_landmarks:
        for a, b in HAND_CONNECTIONS:
            pa = _to_pixel(hand[a], w, h)
            pb = _to_pixel(hand[b], w, h)
            cv2.line(frame_bgr, pa, pb, LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)
        for landmark in hand:
            p = _to_pixel(landmark, w, h)
            cv2.circle(frame_bgr, p, HAND_JOINT_RADIUS, JOINT_COLOR, -1, cv2.LINE_AA)

    return frame_bgr


def draw_face_box(frame_bgr, bbox):
    """Draws a bounding box rectangle around a detected face."""
    if bbox is None:
        return frame_bgr
    x, y, w, h = bbox
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), FACE_BOX_COLOR, 2, cv2.LINE_AA)
    return frame_bgr