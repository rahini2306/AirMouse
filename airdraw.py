"""Use your webcam and one hand as a macOS mouse."""
import argparse
import ctypes as ct
from math import dist, exp
from pathlib import Path
import sys
from time import monotonic_ns

import cv2
import mediapipe as mp

WIDTH, HEIGHT = 960, 540
# The inset camera area maps to the entire main display.
LEFT, RIGHT, TOP, BOTTOM = 155, 805, 95, 455
MOVE, DOWN, UP, DRAG = 5, 1, 2, 6
PINCH_CLOSE, PINCH_OPEN = 0.28, 0.43
DRAG_DELAY_MS = 100


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


def hand_anchor(points):
    return tuple((2 * index + middle + ring) / 4
                 for index, middle, ring in zip(points[5], points[9], points[13]))


class GestureMouse:
    def __init__(self, mouse):
        self.mouse = mouse
        self.pressed = False
        self.reset()

    def reset(self):
        if self.pressed:
            self.mouse.send(UP, self.position)
        self.pressed = False
        self.ready = False
        self.position = None
        self.previous_target = None
        self.previous_time = None
        self.speed = 0.0
        self.pinch_started = None
        self.aim_locked = False

    def update(self, points, timestamp):
        palm = max(dist(points[0], points[9]), 1)
        pinch = dist(points[4], points[8]) / palm
        # Require an open pinch after startup or tracking loss.
        if pinch >= PINCH_OPEN:
            self.ready = True
            self.aim_locked = False
            if self.pressed:
                self.mouse.send(UP, self.position)
                self.pressed = False
                self.previous_target = None
                return
        if not self.ready:
            return

        if not self.pressed and pinch < PINCH_CLOSE and self.position is not None:
            self.mouse.send(DOWN, self.position)
            self.pressed = True
            self.pinch_started = timestamp

        if self.pressed:
            if timestamp - self.pinch_started < DRAG_DELAY_MS:
                self.previous_target = None
                return
        else:
            # Hold the last aim point while the fingers close for a click.
            self.aim_locked = self.aim_locked or pinch < 0.36
            if self.aim_locked:
                self.previous_target = None
                return

        # Average three palm-side knuckles to reduce noise from one landmark.
        x, y = hand_anchor(points)
        target = (max(0, min(1, (x - LEFT) / (RIGHT - LEFT))),
                  max(0, min(1, (y - TOP) / (BOTTOM - TOP))))
        if self.previous_target is None:
            dt = 1 / 30
            self.speed = 0.0
        else:
            dt = max(0.001, min(0.1, (timestamp - self.previous_time) / 1000))
            raw_speed = dist(target, self.previous_target) / dt
            self.speed += (1 - exp(-dt / 0.06)) * (raw_speed - self.speed)
        self.previous_target = target
        self.previous_time = timestamp

        if self.position is not None:
            # Strong smoothing for fine movements, less smoothing for sweeps.
            response = 1 - exp(-dt * (8 + 44 * min(self.speed / 1.4, 1)))
            target = tuple(new if new in (0, 1) else old + response * (new - old)
                           for old, new in zip(self.position, target))
            if dist(target, self.position) < 0.0008 and not any(
                new in (0, 1) and new != old for old, new in zip(self.position, target)
            ):
                return
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
                    points = [(p.x * WIDTH, p.y * HEIGHT) for p in hands[0]]
                    for point in points:
                        cv2.circle(frame, tuple(map(round, point)), 3, (110, 235, 160), -1)
                    cv2.circle(frame, tuple(map(round, hand_anchor(points))), 9, (0, 220, 255), 2)
                    if enabled:
                        control.update(points, timestamp)
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
