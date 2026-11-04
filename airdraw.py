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
CLICK_STATE = 1
DOUBLE_CLICK_MS = 350
PRESS_DEPTH, RELEASE_DEPTH = 0.18, 0.08
HOLD_MS = 180
CALIBRATION_MS = 220


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


class GestureMouse:
    def __init__(self, mouse):
        self.mouse = mouse
        self.taps_enabled = False
        self.pressed = False
        self.reset()

    def reset(self):
        if self.pressed:
            self.mouse.send(UP, self.position, self.clicks)
        self.pressed = False
        self.position = None
        self.previous_target = None
        self.previous_time = None
        self.speed = 0.0
        self.last_click = None
        self.baseline = None
        self.depth = None
        self.calibration_start = None
        self.press_since = None
        self.press_target = None

    @property
    def ready(self):
        return self.baseline is not None

    def update(self, points, tip_depth, timestamp):
        x, y = points[8]
        target = (max(0, min(1, (x - LEFT) / (RIGHT - LEFT))),
                  max(0, min(1, (y - TOP) / (BOTTOM - TOP))))
        palm = dist(points[0], points[9])
        if self.taps_enabled and palm >= 18:
            if self.update_press(tip_depth / palm, target, timestamp):
                return
        elif self.taps_enabled:
            # A small or edge-on palm makes depth unreliable. Keep aiming
            # from the fingertip while recalibrating depth on the next good frame.
            if self.pressed:
                self.mouse.send(UP, self.position, self.clicks)
                self.pressed = False
            self.baseline = self.depth = self.press_since = self.last_click = None

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
        if self.pressed:
            self.dragged = True
        self.position = target


    def update_press(self, sample, target, timestamp):
        if self.depth is None:
            self.depth = sample
            self.calibration_start = timestamp
        self.depth += 0.55 * (sample - self.depth)
        if not self.ready:
            if timestamp - self.calibration_start >= CALIBRATION_MS:
                self.baseline = self.depth
            return False

        push = self.baseline - self.depth
        if self.pressed:
            if push <= RELEASE_DEPTH:
                self.mouse.send(UP, self.position, self.clicks)
                self.last_click = ((timestamp, self.position)
                                   if not self.dragged and self.clicks == 1 else None)
                self.pressed = False
                self.press_since = None
                return True
            return timestamp - self.down_time < HOLD_MS

        if self.position is None or push < PRESS_DEPTH:
            self.press_since = None
            rate = 0.22 if self.depth > self.baseline else 0.04
            self.baseline += rate * (self.depth - self.baseline)
            return False

        if self.press_since is None:
            self.press_since = timestamp
            self.press_target = target
            return False
        if dist(target, self.press_target) > 0.018:
            # A sideways sweep is aiming, even when the depth estimate drifts.
            self.press_since = None
            self.baseline = self.depth
            return False
        if timestamp - self.press_since < 25:
            return False

        self.clicks = 1
        if self.last_click is not None:
            when, where = self.last_click
            if timestamp - when <= DOUBLE_CLICK_MS and dist(self.position, where) < 0.01:
                self.clicks = 2
        self.mouse.send(DOWN, self.position, self.clicks)
        self.pressed = True
        self.dragged = False
        self.down_time = timestamp
        self.previous_target = None
        return True


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
                        hand = hands[0]
                        try:
                            control.update(points, (hand[8].z - hand[5].z) * WIDTH, timestamp)
                        except RuntimeError as error:
                            print(error, file=sys.stderr)
                            control = None
                            mouse_status = "Cursor error. See Terminal."

                preview = render(frame, points, (LEFT, TOP, RIGHT, BOTTOM),
                                 control, mouse_status, fps)
                cv2.imshow("AirMouse", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty("AirMouse", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("r") and control is not None:
                    control.reset()
                if key == ord("t") and control is not None:
                    control.reset()
                    control.taps_enabled = not control.taps_enabled
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
