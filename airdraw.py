"""Use your webcam and one hand as a macOS mouse."""
import argparse
import ctypes as ct
from math import dist, hypot
from pathlib import Path
import sys
from time import monotonic_ns

import cv2
import mediapipe as mp

WIDTH, HEIGHT = 960, 540
# The inset camera area maps to the entire main display.
LEFT, RIGHT, TOP, BOTTOM = 155, 805, 95, 455
MOVE, DOWN, UP, DRAG = 5, 1, 2, 6


class Point(ct.Structure):
    _fields_ = [("x", ct.c_double), ("y", ct.c_double)]


class Size(ct.Structure):
    _fields_ = [("width", ct.c_double), ("height", ct.c_double)]


class Rect(ct.Structure):
    _fields_ = [("origin", Point), ("size", Size)]


class MacMouse:
    def __init__(self):
        self.cg = ct.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        self.cf = ct.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        self.cg.CGPreflightPostEventAccess.restype = ct.c_bool
        if not self.cg.CGPreflightPostEventAccess():
            raise SystemExit(
                "Allow the app you launched from (Terminal or VS Code) in System Settings > "
                "Privacy & Security > Accessibility, then reopen it and run again."
            )
        self.cg.CGMainDisplayID.restype = ct.c_uint32
        self.cg.CGDisplayBounds.argtypes = [ct.c_uint32]
        self.cg.CGDisplayBounds.restype = Rect
        self.bounds = self.cg.CGDisplayBounds(self.cg.CGMainDisplayID())
        self.cg.CGEventCreateMouseEvent.argtypes = [ct.c_void_p, ct.c_uint32, Point, ct.c_uint32]
        self.cg.CGEventCreateMouseEvent.restype = ct.c_void_p
        self.cg.CGEventPost.argtypes = [ct.c_uint32, ct.c_void_p]
        self.cg.CGEventPost.restype = None
        self.cf.CFRelease.argtypes = [ct.c_void_p]
        self.cf.CFRelease.restype = None

    def send(self, kind, position):
        x, y = position
        bounds = self.bounds
        point = Point(bounds.origin.x + x * (bounds.size.width - 1),
                      bounds.origin.y + y * (bounds.size.height - 1))
        event = self.cg.CGEventCreateMouseEvent(None, kind, point, 0)
        if not event:
            raise RuntimeError("macOS could not create a mouse event")
        try:
            self.cg.CGEventPost(0, event)
        finally:
            self.cf.CFRelease(event)


class GestureMouse:
    def __init__(self, mouse):
        self.mouse = mouse
        self.position = None
        self.pressed = False
        self.ready = False

    def reset(self):
        if self.pressed:
            self.mouse.send(UP, self.position)
        self.pressed = False
        self.position = None
        self.ready = False

    def update(self, points):
        palm = max(dist(points[0], points[9]), 1)
        pinch = dist(points[4], points[8]) / palm
        # After startup or tracking loss, require an open pinch before resuming.
        if pinch >= 0.43:
            self.ready = True
        if not self.ready:
            return

        # Freeze on the frame a pinch begins or ends so clicking does not
        # shift the cursor as the fingertip curls.
        if self.pressed and pinch >= 0.43:
            self.mouse.send(UP, self.position)
            self.pressed = False
            return
        if not self.pressed and pinch < 0.28:
            if self.position is not None:
                self.mouse.send(DOWN, self.position)
                self.pressed = True
            return

        # The index knuckle is steadier than the fingertip during a pinch.
        x, y = points[5]
        target = (max(0, min(1, (x - LEFT) / (RIGHT - LEFT))),
                  max(0, min(1, (y - TOP) / (BOTTOM - TOP))))
        if self.position is not None:
            old_x, old_y = self.position
            dx, dy = target[0] - old_x, target[1] - old_y
            if hypot(dx, dy) < 0.003 and all(0 < value < 1 for value in target):
                return
            target = tuple(new if new in (0, 1) else 0.48 * new + 0.52 * old
                           for old, new in zip(self.position, target))
        self.mouse.send(DRAG if self.pressed else MOVE, target)
        self.position = target


def run(camera, model):
    control = GestureMouse(MacMouse())
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    capture = cv2.VideoCapture(camera)
    enabled = False
    timestamp = -1
    try:
        if not capture.isOpened():
            raise SystemExit("Could not open the webcam. Check Camera permissions or try --camera 1.")
        with vision.HandLandmarker.create_from_options(options) as tracker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise SystemExit("The webcam stopped returning frames.")
                frame = cv2.flip(cv2.resize(frame, (WIDTH, HEIGHT)), 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = max(timestamp + 1, monotonic_ns() // 1_000_000)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hands = tracker.detect_for_video(image, timestamp).hand_landmarks

                if hands:
                    points = [(int(p.x * WIDTH), int(p.y * HEIGHT)) for p in hands[0]]
                    for point in points:
                        cv2.circle(frame, point, 3, (110, 235, 160), -1)
                    cv2.circle(frame, points[5], 9, (0, 220, 255), 2)
                    if enabled:
                        control.update(points)
                else:
                    control.reset()

                status = "PAUSED"
                if enabled:
                    status = ("SHOW YOUR HAND" if not hands else "OPEN PINCH TO RESUME" if not control.ready
                              else "DRAGGING" if control.pressed else "POINTER ACTIVE")
                cv2.rectangle(frame, (LEFT, TOP), (RIGHT, BOTTOM), (110, 235, 160), 2)
                cv2.rectangle(frame, (0, 0), (WIDTH, 76), (30, 25, 20), -1)
                cv2.putText(frame, f"AirMouse  |  {status}", (18, 31), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (110, 235, 160), 2, cv2.LINE_AA)
                cv2.putText(frame, "SPACE start/pause  |  Pinch click or drag  |  Q quit",
                            (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.imshow("AirMouse", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty("AirMouse", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(" "):
                    control.reset()
                    enabled = not enabled
    finally:
        try:
            control.reset()
        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("hand_landmarker.task"))
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("AirMouse currently supports macOS.")
    if not args.model.is_file():
        parser.error("Hand Landmarker model missing. See README.md.")
    run(args.camera, args.model)
