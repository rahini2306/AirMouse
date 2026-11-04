"""Render AirMouse's camera overlay and diagnostics."""
import cv2
import numpy as np

AMBER = (59, 174, 255)
GOLD = (115, 212, 255)
INK = (17, 18, 22)
WHITE = (232, 242, 251)
MUTED = (144, 153, 169)
BORDER = (66, 68, 76)
FINGERS = ((0, 1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
           (13, 14, 15, 16), (17, 18, 19, 20))
PALM = (0, 5, 9, 13, 17, 0)


def label(frame, value, x, y, size=0.43, color=MUTED, weight=1):
    cv2.putText(frame, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, weight, cv2.LINE_AA)


def draw_hand(feed, points, clicked):
    joints = np.rint(np.clip(points, -4096, 4096)).astype(np.int32)
    wrist = joints[0]
    light = np.zeros_like(feed)

    for chain in (*FINGERS, PALM):
        path = joints[list(chain)]
        cv2.polylines(light, [path], False, AMBER, 6, cv2.LINE_AA)
        cv2.polylines(feed, [path], False, AMBER, 1, cv2.LINE_AA)

    for finger in FINGERS:
        tip = joints[finger[-1]]
        bend = (tip + wrist) / 2 + (0, 24)
        t = np.linspace(0, 1, 24)[:, None]
        arc = np.rint((1 - t)**2 * wrist + 2 * (1 - t) * t * bend + t**2 * tip).astype(np.int32)
        cv2.polylines(feed, [arc], False, (49, 122, 181), 1, cv2.LINE_AA)
        cv2.circle(light, tuple(tip), 8, GOLD, -1, cv2.LINE_AA)

    glow = cv2.GaussianBlur(light, (13, 13), 0)
    cv2.addWeighted(feed, 1, glow, 0.48, 0, dst=feed)
    for x, y in joints:
        cv2.circle(feed, (x, y), 2, GOLD, -1, cv2.LINE_AA)
    for finger in FINGERS:
        cv2.circle(feed, tuple(joints[finger[-1]]), 4, WHITE, -1, cv2.LINE_AA)
    target = tuple(joints[8])
    cv2.circle(feed, target, 15 if clicked else 11, WHITE if clicked else AMBER, 1, cv2.LINE_AA)


def render(frame, points, bounds, enabled, ready, clicked, fps):
    height, width = frame.shape[:2]
    canvas = np.full((height + 94, width + 288, 3), INK, dtype=np.uint8)
    feed = canvas[56:56 + height, 12:12 + width]
    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    cv2.addWeighted(gray, 0.54, frame, 0.24, 0, dst=feed)

    left, top, right, bottom = bounds
    for x, dx in ((left, 1), (right, -1)):
        for y, dy in ((top, 1), (bottom, -1)):
            cv2.line(feed, (x, y), (x + 22 * dx, y), AMBER, 1, cv2.LINE_AA)
            cv2.line(feed, (x, y), (x, y + 22 * dy), AMBER, 1, cv2.LINE_AA)
    label(feed, "CURSOR AREA", left + 7, top - 10, 0.34, GOLD)
    if points is not None:
        draw_hand(feed, points, clicked)

    cv2.rectangle(canvas, (11, 55), (width + 12, height + 56), BORDER, 1)
    cv2.rectangle(canvas, (12, 15), (16, 39), AMBER, -1)
    label(canvas, "AirMouse", 27, 36, 0.77, WHITE, 2)
    label(canvas, "VISION INTERFACE / 01", 166, 35, 0.40)
    label(canvas, "CAMERA LIVE", width - 92, 34, 0.39, GOLD)

    x = width + 25
    cv2.rectangle(canvas, (x, 55), (width + 276, height + 56), BORDER, 1)
    label(canvas, "01   TRACKING STATUS", x + 15, 82, 0.44, AMBER)
    status = "PAUSED"
    if enabled:
        status = ("SHOW YOUR HAND" if points is None else "CLICK REGISTERED" if clicked
                  else "POINTER ACTIVE" if ready else "OPEN YOUR HAND")
    cv2.circle(canvas, (x + 19, 107), 3, AMBER if enabled else MUTED, -1, cv2.LINE_AA)
    label(canvas, status, x + 30, 112, 0.49, WHITE)
    label(canvas, f"LANDMARKS    {len(points) if points is not None else 0:02d} / 21", x + 15, 148)
    label(canvas, f"FRAME RATE   {fps:04.1f} FPS", x + 15, 171)
    cv2.line(canvas, (x + 15, 195), (width + 261, 195), BORDER, 1)
    label(canvas, "02   GESTURE GUIDE", x + 15, 224, 0.44, AMBER)
    for y, title, hint in (
        (264, "AIM", "Open palm / index fingertip"),
        (324, "CLICK", "Tuck thumb toward palm"),
        (384, "REARM", "Extend thumb again"),
        (444, "PAUSE", "Close hand or lower it"),
    ):
        label(canvas, title, x + 15, y, 0.45, WHITE)
        label(canvas, hint, x + 15, y + 22, 0.39)
    label(canvas, "SPACE  START / PAUSE       Q  QUIT", 14, height + 81, 0.43, GOLD)
    return canvas
