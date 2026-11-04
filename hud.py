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


def start_button(width):
    return width + 18, 434, width + 237, 471


def render(frame, points, bounds, control, mouse_status, fps, left_finger):
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
        if control is not None and control.active_finger == left_finger:
            cv2.circle(feed, tuple(joints[8]), 11, (255, 255, 255), 2, cv2.LINE_AA)

    label(canvas, "AirMouse", 16, 32, 0.72, INK, 2)
    label(canvas, f"LIVE CAMERA  /  {fps:.0f} FPS", 150, 30, 0.4)
    x = width + 18
    status = "Paused"
    if control is not None:
        status = ("Show a hand" if points is None else "Pinch active" if control.active_finger is not None
                  else "Aiming" if control.ready else "Open your fingers")
    elif mouse_status:
        status = mouse_status
    label(canvas, status, x, 30, 0.46, BLUE, 1)
    label(canvas, "Gesture guide", x, 81, 0.62, INK, 2)
    for y, title, hint in (
        (127, "Left-click", "Thumb + index, then release"),
        (187, "Double-click", "Two quick index pinches"),
        (247, "Right-click", "Thumb + middle, then release"),
        (307, "Drag", "Hold index pinch and move"),
        (367, "Scroll", "Hold ring pinch and move up/down"),
    ):
        cv2.circle(canvas, (x + 9, y - 5), 6, BLUE, -1, cv2.LINE_AA)
        label(canvas, title, x + 27, y, 0.5, INK, 1)
        label(canvas, hint, x + 27, y + 22, 0.38)
        cv2.line(canvas, (x, y + 40), (width + 236, y + 40), (231, 225, 219), 1)
    if control is None:
        left, top, right, bottom = start_button(width)
        cv2.rectangle(canvas, (left, top), (right, bottom), BLUE, -1)
        label(canvas, "Start cursor", left + 14, top + 24, 0.51, (255, 255, 255))
    label(canvas, "SPACE Start / pause    R Reset    Q Quit", 16, height + 72, 0.39, INK)
    if mouse_status == "Accessibility needed":
        label(canvas, "Enable Terminal or VS Code in macOS Accessibility, then retry.", 395,
              height + 72, 0.43, BLUE)
    return canvas
