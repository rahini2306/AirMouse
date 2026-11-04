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
| Move | Open your hand and aim with your index fingertip |
| Click | Keep four fingers extended and tuck your thumb toward your palm |
| Double-click | Tuck, reopen your thumb, and tuck again quickly at the same spot |
| Temporarily stop sending events | Close your hand or lower it out of view |
| Quit | Focus the preview and press **Q** or **Esc** |

The app starts paused. Face your palm toward the camera and spread your thumb to enable the pointer. The cursor holds its aim during a thumb tuck and clicks once until you open your thumb again. Cursor control covers the main display.

Gesture design inspired by [Gesture Controlled Virtual Mouse](https://github.com/xenon-19/Gesture-Controlled-Virtual-Mouse).
