# AirDraw

Draw on a virtual canvas using a webcam and one hand. AirDraw runs locally with Python, OpenCV, and MediaPipe Hand Landmarker.

## Run on macOS

Use Python 3.12. If it is missing and you use Homebrew, install it with `brew install python@3.12`. Open Terminal in the extracted AirDraw folder and run:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
curl -fL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task
python airdraw.py
```

Allow camera access when macOS prompts. If necessary, check System Settings → Privacy & Security → Camera. Use `python airdraw.py --camera 1` to try a second camera.

## Controls

- Move your index fingertip below the toolbar. Pinch your thumb and index finger to draw and release to lift the pen.
- Hold your unpinched index fingertip over a toolbar button for about two-thirds of a second. The bar shows selection progress.
- Choose among four colors, erase, undo, clear, or export a clean PNG into the current directory. Undo supports the last 20 edits, including clear.
- Press **U** for undo, **S** to save, or **Q** / **Esc** to quit.

## Design and demo

The pinch threshold scales with palm size, and opening and closing use different thresholds to reduce flicker. AirDraw smooths the cursor and ends a stroke if tracking drops, so reacquiring a hand does not connect two strokes. After a drop, release and pinch again to resume.

Draw two strokes, move your hand out of frame, and return elsewhere. Release and pinch to start a third stroke. Show undo, then export the drawing.

## VS Code

Open `AirDraw.code-workspace`. After setup, select `.venv/bin/python` using **Python: Select Interpreter** if needed, then choose **Run AirDraw** in Run and Debug.
