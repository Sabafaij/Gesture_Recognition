# Real-Time Hand Gesture Recognition (Audio Feedback)
This project provides a full starter implementation for a webcam-based hand gesture recognition system designed for visually impaired users.

## Features
- Real-time hand tracking using MediaPipe (21 landmarks)
- Landmark normalization (63 features)
- MLP classifier for gesture prediction
- Offline speech output using pyttsx3
- Interactive data collection and retraining flow

## Project Structure
- `src/collect_data.py`: collect labeled landmark samples
- `src/train_model.py`: train MLP model from collected CSV data
- `src/realtime_inference.py`: run live prediction + spoken output
- `config/labels.json`: supported gesture classes
- `data/raw/keypoints.csv`: collected dataset (generated)
- `models/gesture_mlp.pkl`: trained classifier (generated)

## Setup
1. Create and activate virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Recommended Python version: `3.11` or `3.12` (for stable package compatibility).

## Migration Notes (MediaPipe Tasks API)
The project has been migrated from legacy `mp.solutions.hands` usage to the current MediaPipe Tasks API (`mp.tasks.vision.HandLandmarker`).

### What changed
- `collect_data.py` and `realtime_inference.py` now use the Tasks hand landmarker pipeline.
- `src/utils.py` now provides:
  - hand landmarker initialization via Tasks API
  - frame detection helper
  - landmark extraction/normalization helpers
  - manual landmark drawing for visualization
- The hand landmarker model asset is now managed as:
  - `models/hand_landmarker.task`

### Model asset behavior
- On first run, the project auto-downloads `hand_landmarker.task` if missing.
- The downloaded model is reused on subsequent runs.

### Why this migration was required
- Newer MediaPipe builds no longer expose `mp.solutions` in the same way, causing runtime failures with older code paths.
- Tasks API is the current supported interface for hand landmark detection.

## Usage
### 1) Collect data for each label
Run this multiple times, one label at a time:

`python src/collect_data.py --label A --samples 250`

Press:
- `s` to start/pause recording
- `q` to quit

### 2) Train model
`python src/train_model.py`

### 3) Run real-time inference with voice
`python src/realtime_inference.py --threshold 0.65`

## Notes
- Ensure your webcam works and has good lighting.
- Collect balanced samples for each class for better accuracy.
- You can edit `config/labels.json` to add/remove gesture labels.
