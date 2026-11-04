"""Use your webcam and one hand as a macOS mouse."""
import argparse
import ctypes as ct
from math import dist, exp
from pathlib import Path
import sys
from time import monotonic_ns

import cv2
import mediapipe as mp

from hud import render

WIDTH, HEIGHT = 960, 540
# The inset camera area maps to the entire main display.
LEFT, RIGHT, TOP, BOTTOM = 155, 805, 95, 455
MOVE, DOWN, UP = 5, 1, 2
CLICK_STATE = 1
DOUBLE_CLICK_MS = 350
THUMB_OPEN, THUMB_TUCKED = 0.32, 0.08
POSE_HOLD_MS = 55


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
        self.cg.CGEventSetIntegerValueField.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_int64]
        self.cg.CGEventSetIntegerValueField.restype = None
        self.cg.CGEventPost.argtypes = [ct.c_uint32, ct.c_void_p]
        self.cg.CGEventPost.restype = None
        self.cf.CFRelease.argtypes = [ct.c_void_p]
        self.cf.CFRelease.restype = None

    def send(self, kind, position, clicks=1):
        x, y = position
        bounds = self.bounds
        point = Point(bounds.origin.x + x * (bounds.size.width - 1),
                      bounds.origin.y + y * (bounds.size.height - 1))
        event = self.cg.CGEventCreateMouseEvent(None, kind, point, 0)
        click_state = clicks if kind in (DOWN, UP) else None
        self.post(event, click_state)

    def post(self, event, click_state=None):
        if not event:
            raise RuntimeError("macOS could not create a mouse event")
        try:
            if click_state is not None:
                self.cg.CGEventSetIntegerValueField(event, CLICK_STATE, click_state)
            self.cg.CGEventPost(0, event)
        finally:
            self.cf.CFRelease(event)


def palm_pose(points):
    wrist, index, middle, pinky = (points[i] for i in (0, 5, 9, 17))
    forward = (middle[0] - wrist[0], middle[1] - wrist[1])
    thumb_side = (index[0] - pinky[0], index[1] - pinky[1])
    palm_length = dist(wrist, middle)
    palm_width = dist(index, pinky)
    if palm_length < 12 or palm_width < 10:
        return False, 0.0

    def projection(tip, joint, axis):
        return sum((end - start) * direction for end, start, direction in zip(tip, joint, axis))

    fingers_open = all(projection(points[tip], points[tip - 2], forward) > 0.07 * palm_length**2
                       for tip in (8, 12, 16, 20))
    thumb_extension = projection(points[4], index, thumb_side) / palm_width**2
    return fingers_open, thumb_extension


class GestureMouse:
    def __init__(self, mouse):
        self.mouse = mouse
        self.reset()

    def reset(self):
        self.ready = False
        self.position = None
        self.previous_target = None
        self.previous_time = None
        self.speed = 0.0
        self.last_click = None
        self.open_since = None
        self.tuck_since = None
        self.clicked = False

    def click(self, timestamp):
        clicks = 1
        if self.last_click is not None:
            click_time, click_position = self.last_click
            if timestamp - click_time <= DOUBLE_CLICK_MS and dist(self.position, click_position) < 0.007:
                clicks = 2
        self.mouse.send(DOWN, self.position, clicks)
        self.mouse.send(UP, self.position, clicks)
        self.last_click = (timestamp, self.position) if clicks == 1 else None

    def update(self, points, timestamp):
        fingers_open, thumb = palm_pose(points)
        if not fingers_open:
            self.reset()
            return

        if thumb >= THUMB_OPEN:
            self.tuck_since = None
            if self.open_since is None:
                self.open_since = timestamp
            if timestamp - self.open_since >= POSE_HOLD_MS:
                self.ready = True
                self.clicked = False
        else:
            self.open_since = None

        if not self.ready:
            return
        if thumb < 0.20:
            # Freeze the last aim point before the thumb reaches click position.
            self.previous_target = None
            if thumb <= THUMB_TUCKED and not self.clicked and self.position is not None:
                if self.tuck_since is None:
                    self.tuck_since = timestamp
                elif timestamp - self.tuck_since >= POSE_HOLD_MS:
                    self.click(timestamp)
                    self.clicked = True
            else:
                self.tuck_since = None
            return
        self.tuck_since = None
        if self.clicked:
            return

        x, y = points[8]
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
        self.mouse.send(MOVE, target)
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
    fps = 0.0
    try:
        if not capture.isOpened():
            raise SystemExit("Could not open the webcam. Check Camera permissions or try --camera 1.")
        with vision.HandLandmarker.create_from_options(options) as tracker:
            cv2.namedWindow("AirMouse", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("AirMouse", WIDTH + 288, HEIGHT + 94)
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise SystemExit("The webcam stopped returning frames.")
                frame = cv2.flip(cv2.resize(frame, (WIDTH, HEIGHT)), 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                previous_timestamp = timestamp
                timestamp = max(timestamp + 1, monotonic_ns() // 1_000_000)
                if previous_timestamp >= 0:
                    sample = 1000 / (timestamp - previous_timestamp)
                    fps = 0.9 * fps + 0.1 * sample if fps else sample
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hands = tracker.detect_for_video(image, timestamp).hand_landmarks

                points = [(p.x * WIDTH, p.y * HEIGHT) for p in hands[0]] if hands else None
                if enabled and points is not None:
                    control.update(points, timestamp)
                elif points is None:
                    control.reset()

                preview = render(frame, points, (LEFT, TOP, RIGHT, BOTTOM),
                                 enabled, control.ready, control.clicked, fps)
                cv2.imshow("AirMouse", preview)
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
