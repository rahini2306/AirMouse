"""Webcam drawing with pinch-to-ink and dwell-to-select."""
import argparse
from datetime import datetime
from math import dist
from pathlib import Path
from time import monotonic_ns

import cv2
import mediapipe as mp
import numpy as np

WIDTH, HEIGHT = 960, 540
BAR = 64
COLORS = ((240, 170, 35), (135, 70, 240), (75, 195, 90), (30, 30, 30))
TOOLS = ("Blue", "Pink", "Green", "Black", "Eraser", "Undo", "Clear", "Save")
CELL = WIDTH // len(TOOLS)
LINKS = [(0, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
for base in (1, 5, 9, 13, 17):
    LINKS.extend((i, i + 1) for i in range(base, base + 3))
LINKS.append((0, 1))


class Canvas:
    def __init__(self):
        self.image = np.full((HEIGHT - BAR, WIDTH, 3), 255, dtype=np.uint8)
        self.history = []
        self.previous = None
        self.tool = 0
        self.notice = ""

    def checkpoint(self):
        self.history.append(self.image.copy())
        if len(self.history) > 20:
            self.history.pop(0)

    def stroke(self, point):
        x, y = point
        point = (x, y - BAR)
        if self.previous is None:
            self.checkpoint()
            self.previous = point
        erasing = self.tool == 4
        color = (255, 255, 255) if erasing else COLORS[self.tool]
        cv2.line(self.image, self.previous, point, color, 28 if erasing else 5, cv2.LINE_AA)
        self.previous = point

    def select(self, slot):
        self.previous = None
        self.notice = ""
        if slot < 5:
            self.tool = slot
        elif slot == 5 and self.history:
            self.image = self.history.pop()
        elif slot == 6 and np.any(self.image != 255):
            self.checkpoint()
            self.image.fill(255)
        elif slot == 7:
            path = Path(f"airdraw-{datetime.now():%Y%m%d-%H%M%S-%f}.png")
            try:
                saved = cv2.imwrite(str(path), self.image)
            except cv2.error:
                saved = False
            self.notice = f"Saved {path.name}" if saved else "Could not save PNG"


class Gesture:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pinched = False
        self.ready = False
        self.smooth = None
        self.hover = None
        self.hover_since = 0
        self.selected = False

    def update(self, points, now, canvas):
        palm = max(dist(points[0], points[9]), 1)
        ratio = dist(points[4], points[8]) / palm
        self.pinched = ratio < (0.43 if self.pinched else 0.30)
        if ratio >= 0.43:
            self.ready = True
        x, y = points[8]
        if y < BAR:
            canvas.previous = None
            self.smooth = None
            if self.pinched:
                self.ready = False
            slot = x // CELL if self.ready and not self.pinched else None
            if slot != self.hover:
                self.hover = slot
                self.hover_since = now
                self.selected = False
            if slot is not None and not self.selected and now - self.hover_since >= 650:
                canvas.select(slot)
                self.selected = True
            return x, y

        self.hover = None
        self.smooth = ((x, y) if self.smooth is None else
                       (0.55 * x + 0.45 * self.smooth[0], 0.55 * y + 0.45 * self.smooth[1]))
        cursor = tuple(map(round, self.smooth))
        if self.ready and self.pinched:
            canvas.stroke(cursor)
        else:
            canvas.previous = None
        return cursor


def label(image, text, position, scale=0.55, color=(240, 240, 240)):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def run(camera, model):
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    canvas, gesture = Canvas(), Gesture()
    capture = cv2.VideoCapture(camera)
    timestamp = -1
    try:
        if not capture.isOpened():
            raise SystemExit("Could not open the webcam. Check camera permissions or try --camera 1.")
        with vision.HandLandmarker.create_from_options(options) as tracker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise SystemExit("The webcam stopped returning frames.")
                frame = cv2.flip(cv2.resize(frame, (WIDTH, HEIGHT)), 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = max(timestamp + 1, monotonic_ns() // 1_000_000)
                result = tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                cursor = None
                if result.hand_landmarks:
                    points = [(min(WIDTH - 1, max(0, int(p.x * WIDTH))),
                               min(HEIGHT - 1, max(0, int(p.y * HEIGHT))))
                              for p in result.hand_landmarks[0]]
                    cursor = gesture.update(points, timestamp, canvas)
                    for start, end in LINKS:
                        cv2.line(frame, points[start], points[end], (100, 255, 175), 2, cv2.LINE_AA)
                else:
                    canvas.previous = None
                    gesture.reset()

                frame[:BAR] = (36, 31, 28)
                frame[BAR:] = cv2.addWeighted(frame[BAR:], 0.24, canvas.image, 0.76, 0)
                for i, name in enumerate(TOOLS):
                    x = i * CELL
                    if i == canvas.tool:
                        cv2.rectangle(frame, (x + 2, 3), (x + CELL - 3, BAR - 4), (200, 200, 200), 1)
                    if i < len(COLORS):
                        cv2.circle(frame, (x + 15, 31), 5, COLORS[i], -1)
                    label(frame, name, (x + 26, 37))
                    if i == gesture.hover:
                        progress = min(1, (timestamp - gesture.hover_since) / 650)
                        cv2.line(frame, (x + 2, BAR - 5), (x + 2 + int((CELL - 5) * progress), BAR - 5),
                                 (100, 255, 175), 3)
                if cursor is not None:
                    cv2.circle(frame, cursor, 7, (35, 145, 40) if gesture.pinched else (20, 20, 20), 2)
                status = canvas.notice or ("Show one hand to start" if cursor is None else
                                           "Pinch to draw | Release to hover | U undo | S save | Q quit")
                cv2.rectangle(frame, (0, HEIGHT - 29), (WIDTH, HEIGHT), (36, 31, 28), -1)
                label(frame, status, (12, HEIGHT - 9))
                cv2.imshow("AirDraw", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty("AirDraw", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("u"), ord("s")):
                    canvas.select(5 if key == ord("u") else 7)
                    gesture.ready = False
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("hand_landmarker.task"))
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("Hand Landmarker model missing. See the download command in README.md.")
    run(args.camera, args.model)
