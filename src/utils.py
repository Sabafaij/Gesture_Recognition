from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import mediapipe as mp
import numpy as np
import urllib.request

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class HandResult:
    feature_vector: np.ndarray
    handedness: str


def ensure_hand_landmarker_model(model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return
    urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, str(model_path))


def create_hand_landmarker(model_path: Path, num_hands: int = 1):
    ensure_hand_landmarker_model(model_path)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def normalize_landmarks(hand_landmarks) -> np.ndarray:
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    points -= points[0]
    max_val = np.max(np.abs(points))
    if max_val > 1e-6:
        points /= max_val
    return points.flatten()


def extract_first_hand(result) -> Optional[HandResult]:
    if not result or not getattr(result, "hand_landmarks", None):
        return None
    hand_landmarks = result.hand_landmarks[0]
    handedness = "Unknown"
    handedness_groups = getattr(result, "handedness", [])
    if handedness_groups and len(handedness_groups[0]) > 0:
        candidate = handedness_groups[0][0]
        handedness = getattr(candidate, "category_name", None) or getattr(candidate, "display_name", None) or "Unknown"
    vector = normalize_landmarks(hand_landmarks)
    return HandResult(feature_vector=vector, handedness=handedness)


def detect_hands(landmarker, frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect(mp_image)


def draw_landmarks(frame: np.ndarray, result) -> np.ndarray:
    if not result or not getattr(result, "hand_landmarks", None):
        return frame
    h, w, _ = frame.shape
    for hand_landmarks in result.hand_landmarks:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
        for px, py in points:
            cv2.circle(frame, (px, py), 3, (0, 120, 255), -1)
    return frame

def put_status_text(frame: np.ndarray, lines: list[str], x: int = 10, y: int = 30) -> np.ndarray:
    for idx, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x, y + idx * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame
