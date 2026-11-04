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
| Move | Move your hand so the yellow marker stays inside the outlined camera area |
| Click | Pinch your thumb to any fingertip and release |
| Double-click | Pinch twice quickly in the same spot |
| Drag | Hold any pinch briefly, move, then release |
| Temporarily stop sending events | Lower your hand out of view |
| Quit | Focus the preview and press **Q** or **Esc** |

The app starts paused. If tracking is lost mid-drag, AirMouse releases the mouse button. When tracking returns, open your pinch before resuming. Cursor control covers the main display.

Gesture design inspired by [Gesture Controlled Virtual Mouse](https://github.com/xenon-19/Gesture-Controlled-Virtual-Mouse).
