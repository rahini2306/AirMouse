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
| Move | Aim with your index fingertip |
| Click | Push the index finger toward the webcam, then pull back |
| Double-click | Tap twice quickly at the same spot |
| Hold / drag | Press toward the webcam, wait briefly, then move sideways before releasing |
| Recalibrate | Focus the preview, keep your hand at its neutral distance, and press **R** |
| Quit | Focus the preview and press **Q** or **Esc** |

Click **Start cursor** to aim without clicks. When you are ready to try tapping, focus the preview and press **T**. Hold your hand in a comfortable pointing position until the status says **Ready**. Move the index finger toward the camera relative to your palm to press. Moving the whole hand forward is less effective because depth is measured relative to the index knuckle. Tracking loss releases any held button. Cursor control covers the main display.

Gesture design inspired by [Gesture Controlled Virtual Mouse](https://github.com/xenon-19/Gesture-Controlled-Virtual-Mouse).
