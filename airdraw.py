"""Use your webcam and one hand as a macOS mouse."""
import argparse
import ctypes as ct
from math import dist, exp
from pathlib import Path
import sys
from time import monotonic_ns

import cv2
import mediapipe as mp

from hud import render, start_button

WIDTH, HEIGHT = 960, 540
# The inset camera area maps to the entire main display.
LEFT, RIGHT, TOP, BOTTOM = 155, 805, 95, 455
MOVE, DOWN, UP, DRAG = 5, 1, 2, 6
RIGHT_DOWN, RIGHT_UP = 3, 4
CLICK_STATE = 1
INDEX, MIDDLE, RING = 8, 12, 16
DOUBLE_CLICK_MS = 350
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
        self.cg.CGWarpMouseCursorPosition.argtypes = [Point]
        self.cg.CGWarpMouseCursorPosition.restype = ct.c_int32
        self.cg.CGMainDisplayID.restype = ct.c_uint32
        self.cg.CGDisplayBounds.argtypes = [ct.c_uint32]
        self.cg.CGDisplayBounds.restype = Rect
        self.bounds = self.cg.CGDisplayBounds(self.cg.CGMainDisplayID())
        self.cg.CGEventCreateMouseEvent.argtypes = [ct.c_void_p, ct.c_uint32, Point, ct.c_uint32]
        self.cg.CGEventCreateMouseEvent.restype = ct.c_void_p
        self.cg.CGEventSetIntegerValueField.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_int64]
        self.cg.CGEventSetIntegerValueField.restype = None
        self.cg.CGEventCreateScrollWheelEvent2.argtypes = [
            ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_int32, ct.c_int32, ct.c_int32
        ]
        self.cg.CGEventCreateScrollWheelEvent2.restype = ct.c_void_p
        self.cg.CGEventPost.argtypes = [ct.c_uint32, ct.c_void_p]
        self.cg.CGEventPost.restype = None
        self.cf.CFRelease.argtypes = [ct.c_void_p]
        self.cf.CFRelease.restype = None

    def send(self, kind, position, clicks=1):
        x, y = position
        bounds = self.bounds
        point = Point(bounds.origin.x + x * (bounds.size.width - 1),
                      bounds.origin.y + y * (bounds.size.height - 1))
        if kind == MOVE:
            error = self.cg.CGWarpMouseCursorPosition(point)
            if error:
                raise RuntimeError(f"macOS could not move the cursor (Core Graphics error {error}).")
        button = int(kind in (RIGHT_DOWN, RIGHT_UP))
        event = self.cg.CGEventCreateMouseEvent(None, kind, point, button)
        click_state = clicks if kind in (DOWN, UP, RIGHT_DOWN, RIGHT_UP) else None
        self.post(event, click_state)

    def scroll(self, pixels):
        self.post(self.cg.CGEventCreateScrollWheelEvent2(None, 0, 1, pixels, 0, 0))

    def post(self, event, click_state=None):
        if not event:
            raise RuntimeError("macOS could not create a mouse event")
        try:
            if click_state is not None:
                self.cg.CGEventSetIntegerValueField(event, CLICK_STATE, click_state)
            self.cg.CGEventPost(0, event)
        finally:
            self.cf.CFRelease(event)


def hand_anchor(points):
    return tuple((2 * index + middle + ring) / 4
                 for index, middle, ring in zip(points[5], points[9], points[13]))


class GestureMouse:
    def __init__(self, mouse):
        self.mouse = mouse
        self.active_finger = None
        self.reset()

    def reset(self):
        if self.active_finger == INDEX:
            self.mouse.send(UP, self.position, self.clicks)
        self.active_finger = None
        self.ready = False
        self.position = None
        self.previous_target = None
        self.previous_time = None
        self.speed = 0.0
        self.last_click = None
        self.clicks = 1

    def begin_pinch(self, finger, y, timestamp):
        self.active_finger = finger
        self.pinch_started = timestamp
        self.previous_target = None
        self.dragged = False
        if finger == INDEX:
            self.clicks = 1
            if self.last_click is not None:
                click_time, click_position = self.last_click
                if timestamp - click_time <= DOUBLE_CLICK_MS and dist(self.position, click_position) < 0.007:
                    self.clicks = 2
            self.mouse.send(DOWN, self.position, self.clicks)
        else:
            self.last_click = None
            self.scroll_y = y

    def finish_pinch(self, timestamp):
        if self.active_finger == INDEX:
            self.mouse.send(UP, self.position, self.clicks)
            self.last_click = ((timestamp, self.position)
                               if not self.dragged and self.clicks == 1 else None)
        elif self.active_finger == MIDDLE:
            self.mouse.send(RIGHT_DOWN, self.position)
            self.mouse.send(RIGHT_UP, self.position)
        self.active_finger = None
        self.ready = False
        self.previous_target = None

    def update(self, points, timestamp):
        palm = max(dist(points[0], points[9]), 1)
        pinches = {finger: dist(points[4], points[finger]) / palm for finger in (INDEX, MIDDLE, RING)}
        x, y = hand_anchor(points)
        if self.active_finger is not None:
            if pinches[self.active_finger] >= PINCH_OPEN:
                self.finish_pinch(timestamp)
                return
            if self.active_finger == RING:
                delta = self.scroll_y - y
                if abs(delta) >= 3:
                    self.mouse.scroll(round(max(-90, min(90, delta * 2))))
                    self.scroll_y = y
                return
            if self.active_finger == MIDDLE or timestamp - self.pinch_started < DRAG_DELAY_MS:
                return
        else:
            ordered = sorted(pinches, key=pinches.get)
            nearest, runner_up = ordered[:2]
            gap = pinches[nearest]
            if gap >= PINCH_OPEN:
                self.ready = True
            if not self.ready:
                return
            if gap < PINCH_CLOSE and pinches[runner_up] - gap >= 0.08 and self.position is not None:
                self.begin_pinch(nearest, y, timestamp)
                return
            # Freeze the pointer as the fingers close, including ambiguous pinches.
            if gap < 0.36:
                self.previous_target = None
                return

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
        if self.active_finger == INDEX:
            self.mouse.send(DRAG, target)
            self.dragged = True
        else:
            self.mouse.send(MOVE, target)
        self.position = target


def run(camera, model):
    control = None
    mouse_status = ""
    start_requested = False
    left, top, right, bottom = start_button(WIDTH)

    def on_preview_click(event, x, y, flags, data):
        nonlocal start_requested
        if (event == cv2.EVENT_LBUTTONUP and control is None
                and left <= x <= right and top <= y <= bottom):
            start_requested = True

    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    capture = cv2.VideoCapture(camera)
    timestamp = -1
    fps = 0.0
    try:
        if not capture.isOpened():
            raise SystemExit("Could not open the webcam. Check Camera permissions or try --camera 1.")
        with vision.HandLandmarker.create_from_options(options) as tracker:
            cv2.namedWindow("AirMouse", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("AirMouse", WIDTH + 256, HEIGHT + 84)
            cv2.setMouseCallback("AirMouse", on_preview_click)
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
                if control is not None:
                    if points is None:
                        control.reset()
                    else:
                        try:
                            control.update(points, timestamp)
                        except RuntimeError as error:
                            print(error, file=sys.stderr)
                            control = None
                            mouse_status = "Cursor error. See Terminal."

                preview = render(frame, points, (LEFT, TOP, RIGHT, BOTTOM),
                                 control, mouse_status, fps, INDEX)
                cv2.imshow("AirMouse", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty("AirMouse", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("r") and control is not None:
                    control.reset()
                if key == ord(" ") or start_requested:
                    start_requested = False
                    if control is not None:
                        control.reset()
                        control = None
                    else:
                        try:
                            control = GestureMouse(MacMouse())
                            mouse_status = ""
                        except SystemExit as error:
                            print(error, file=sys.stderr)
                            mouse_status = "Accessibility needed"
    finally:
        try:
            if control is not None:
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
