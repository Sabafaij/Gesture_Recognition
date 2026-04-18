from __future__ import annotations
import argparse
import time
import cv2
import joblib
import numpy as np
import pyttsx3
from config import MODEL_PATH, LABEL_ENCODER_PATH, HAND_LANDMARKER_TASK
from utils import create_hand_landmarker, detect_hands, extract_first_hand, draw_landmarks, put_status_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time gesture recognition with voice output.")
    parser.add_argument("--camera", default=0, type=int, help="Webcam index")
    parser.add_argument("--threshold", default=0.65, type=float, help="Prediction confidence threshold")
    parser.add_argument("--speak-interval", default=1.2, type=float, help="Minimum seconds between repeated speech")
    return parser.parse_args()


def init_tts() -> pyttsx3.Engine:
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)
    return engine


def main() -> None:
    args = parse_args()
    if not MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError("Train model first. Missing models/gesture_mlp.pkl or models/label_encoder.pkl")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(LABEL_ENCODER_PATH)
    tts = init_tts()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    last_spoken = ""
    last_spoken_ts = 0.0
    with create_hand_landmarker(HAND_LANDMARKER_TASK, num_hands=1) as hand_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            result = detect_hands(hand_landmarker, frame)
            draw_landmarks(frame, result)

            pred_label = "No Gesture"
            confidence = 0.0
            hand = extract_first_hand(result)
            if hand is not None:
                probs = model.predict_proba(hand.feature_vector.reshape(1, -1))[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                if confidence >= args.threshold:
                    pred_label = str(encoder.inverse_transform([pred_idx])[0])
                    now = time.time()
                    if pred_label != last_spoken or (now - last_spoken_ts) >= args.speak_interval:
                        tts.say(pred_label)
                        tts.runAndWait()
                        last_spoken = pred_label
                        last_spoken_ts = now

            put_status_text(
                frame,
                [
                    f"Prediction: {pred_label}",
                    f"Confidence: {confidence:.2f}",
                    "Press 'q' to exit",
                ],
            )
            cv2.imshow("Real-Time Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
