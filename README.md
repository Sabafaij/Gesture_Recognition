# Real-Time Hand Gesture Recognition (Audio Feedback)
This project implements a webcam-based hand gesture recognition system that converts recognized gestures into offline spoken output. It is designed as an accessibility-focused starter project and runs fully on local hardware.

## Core Features
- Real-time hand landmark detection using MediaPipe Tasks HandLandmarker (21 landmarks)
- Landmark normalization to a 63-dimensional feature vector (21 x 3)
- MLP-based gesture classification
- Offline text-to-speech using `pyttsx3`
- Interactive data collection and retraining workflow
- Local execution with no cloud dependency

## Repository Structure
- `src/collect_data.py` - capture labeled hand landmark samples from webcam
- `src/train_model.py` - train and evaluate the MLP classifier
- `src/realtime_inference.py` - run live prediction with spoken feedback
- `src/utils.py` - MediaPipe Tasks helpers, drawing, and preprocessing
- `src/config.py` - project paths and runtime config
- `config/labels.json` - supported gesture labels
- `data/raw/keypoints.csv` - collected training samples (generated)
- `models/gesture_mlp.pkl` - trained classifier (generated)
- `models/label_encoder.pkl` - fitted label encoder (generated)
- `models/hand_landmarker.task` - MediaPipe model asset (auto-downloaded)

## Requirements
- Python `3.11` or `3.12` recommended
- Webcam
- Windows/Linux/macOS

Install packages:
- `pip install -r requirements.txt`

## MediaPipe Migration Note
The project now uses the MediaPipe **Tasks API** (`mp.tasks.vision.HandLandmarker`) instead of legacy `mp.solutions.hands`.

Why:
- Recent MediaPipe builds changed legacy module availability.
- Tasks API is the maintained path for hand landmark detection.

Behavior:
- The script auto-downloads `hand_landmarker.task` on first run if missing.
- Subsequent runs reuse the local model file from `models/`.

## End-to-End Pipeline
1. Collect labeled gesture samples.
2. Train the MLP model.
3. Run real-time inference with spoken output.

## Usage
### 1) Collect samples
Run per label (repeat for each class in `config/labels.json`):

`python src/collect_data.py --label A --samples 250`

Controls:
- `s` - start/pause capture
- `q` - quit

### 2) Train classifier
`python src/train_model.py`

Outputs:
- `models/gesture_mlp.pkl`
- `models/label_encoder.pkl`

### 3) Run live inference + TTS
`python src/realtime_inference.py --threshold 0.65`

Optional:
- `--camera 0` choose webcam index
- `--speak-interval 1.2` set minimum repeat speech interval

## Tips for Better Accuracy
- Keep lighting consistent during collection and inference.
- Collect balanced samples for every label.
- Capture variation in hand position/orientation per class.
- Start with fewer classes, then expand gradually.

## Troubleshooting
- If inference is unstable, confirm webcam permissions and close other apps using camera.
- If dependencies fail with your default Python, create a virtual environment with Python 3.11/3.12.
- If model files are missing, run training again after collecting samples.
