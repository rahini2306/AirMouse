# AirMouse

Use one hand and your webcam to control the macOS pointer across apps on your main display. MediaPipe tracks hand landmarks, OpenCV shows the camera preview, and macOS Core Graphics sends mouse events.

## Run on macOS

If AirDraw was already set up in this folder, reuse its Python environment and model:

```sh
.venv/bin/python airdraw.py
```

For a fresh setup, use Python 3.12 and run these commands from the project folder:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
curl -fL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task
.venv/bin/python airdraw.py
```

In **System Settings → Privacy & Security → Accessibility**, enable the app you launch from, such as Terminal or Visual Studio Code. Reopen that app after changing permission. Allow camera access when prompted. In VS Code, select **Run AirMouse** in Run and Debug once the environment is ready.

## Controls

| Action | Gesture or key |
| --- | --- |
| Start or pause | Focus the AirMouse preview and press **Space** |
| Move | Move your hand inside the marked camera area |
| Left-click | Pinch thumb and index finger, then release |
| Double-click | Make two quick thumb–index pinches in the same spot |
| Right-click | Pinch thumb and middle finger, then release |
| Drag | Hold a thumb–index pinch briefly, then move your hand |
| Scroll | Hold a thumb–ring finger pinch and move up or down |
| Reset tracking | Focus the preview and press **R** |
| Quit | Focus the preview and press **Q** or **Esc** |

Click **Start cursor** and show one hand with your fingertips separated. If tracking drops during a drag, AirMouse releases the mouse button. Cursor control covers the main display.

Gesture design inspired by [Gesture Controlled Virtual Mouse](https://github.com/xenon-19/Gesture-Controlled-Virtual-Mouse).
