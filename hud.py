"""Live camera view and compact gesture guide."""
import cv2
import numpy as np

GREEN = (55, 245, 66)
RED = (55, 58, 248)
BLUE = (183, 123, 19)
INK = (55, 39, 27)
MUTED = (113, 98, 83)
FINGERS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
           (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))


def label(frame, text, x, y, size=0.46, color=MUTED, weight=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, weight, cv2.LINE_AA)


def render(frame, points, bounds, control, permission_needed, fps):
    height, width = frame.shape[:2]
    canvas = np.full((height + 84, width + 256, 3), (252, 251, 249), dtype=np.uint8)
    feed = canvas[48:48 + height, :width]
    feed[:] = frame
    left, top, right, bottom = bounds
    for x, dx in ((left, 1), (right, -1)):
        for y, dy in ((top, 1), (bottom, -1)):
            cv2.line(feed, (x, y), (x + 14 * dx, y), (240, 240, 240), 1, cv2.LINE_AA)
            cv2.line(feed, (x, y), (x, y + 14 * dy), (240, 240, 240), 1, cv2.LINE_AA)
    if points is not None:
        joints = np.rint(np.clip(points, -4096, 4096)).astype(np.int32)
        for finger in FINGERS:
            cv2.polylines(feed, [joints[list(finger)]], False, GREEN, 2, cv2.LINE_AA)
        for joint in joints:
            cv2.circle(feed, tuple(joint), 3, RED, -1, cv2.LINE_AA)
        if control is not None and control.pressed:
            cv2.circle(feed, tuple(joints[8]), 11, (255, 255, 255), 2, cv2.LINE_AA)

    label(canvas, "AirMouse", 16, 32, 0.72, INK, 2)
    label(canvas, f"LIVE CAMERA  /  {fps:.0f} FPS", 150, 30, 0.4)
    x = width + 18
    status = "Paused"
    if control is not None:
        status = ("Pressing" if control.pressed else "Ready" if control.ready
                  else "Show a hand" if points is None else "Finding neutral depth")
    elif permission_needed:
        status = "Accessibility needed"
    label(canvas, status, x, 30, 0.46, BLUE, 1)
    label(canvas, "Gesture guide", x, 81, 0.62, INK, 2)
    for y, title, hint in (
        (127, "Aim", "Move your index fingertip"),
        (200, "Tap", "Push forward, then release"),
        (273, "Double tap", "Tap twice at the same spot"),
        (346, "Hold / drag", "Press, wait, then move"),
    ):
        cv2.circle(canvas, (x + 9, y - 5), 6, BLUE, -1, cv2.LINE_AA)
        label(canvas, title, x + 27, y, 0.5, INK, 1)
        label(canvas, hint, x + 27, y + 22, 0.38)
        cv2.line(canvas, (x, y + 40), (width + 236, y + 40), (231, 225, 219), 1)
    label(canvas, "SPACE Start / pause    R Reset    Q Quit", 16, height + 72, 0.39, INK)
    if permission_needed:
        label(canvas, "Enable Terminal or VS Code in macOS Accessibility, then retry.", 290,
              height + 72, 0.43, BLUE)
    return canvas
