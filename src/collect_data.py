from __future__ import annotations
import argparse
from pathlib import Path
import csv
import cv2

from config import KEYPOINTS_CSV, RAW_DATA_DIR, HAND_LANDMARKER_TASK, get_supported_labels
from utils import create_hand_landmarker, detect_hands, extract_first_hand, draw_landmarks, put_status_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect hand landmark samples for one label.")
    parser.add_argument("--label", required=True, type=str, help="Gesture label name (e.g., A, STOP, HELP)")
    parser.add_argument("--samples", default=250, type=int, help="Number of samples to collect")
    parser.add_argument("--camera", default=0, type=int, help="Webcam index")
    return parser.parse_args()


def ensure_csv_header(csv_path: Path) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"f{i}" for i in range(63)]
        writer.writerow(header)


def main() -> None:
    args = parse_args()
    label = args.label.strip().upper()
    supported_labels = set(get_supported_labels())
    if label not in supported_labels:
        raise ValueError(f"Label '{label}' not in config/labels.json")

    ensure_csv_header(KEYPOINTS_CSV)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    collected = 0
    collecting = False
    with create_hand_landmarker(HAND_LANDMARKER_TASK, num_hands=1) as hand_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = detect_hands(hand_landmarker, frame)
            draw_landmarks(frame, result)

            hand = extract_first_hand(result)
            if collecting and hand is not None and collected < args.samples:
                with KEYPOINTS_CSV.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([label, *hand.feature_vector.tolist()])
                collected += 1

            status = [
                f"Label: {label}",
                f"Collected: {collected}/{args.samples}",
                "Press 's' start/pause | 'q' quit",
            ]
            if hand is None:
                status.append("No hand detected")
            put_status_text(frame, status)
            cv2.imshow("Collect Gesture Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                collecting = not collecting
            elif key == ord("q") or collected >= args.samples:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {collected} samples for label '{label}' into {KEYPOINTS_CSV}")


if __name__ == "__main__":
    main()
