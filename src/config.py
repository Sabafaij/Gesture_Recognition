from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
CONFIG_DIR = ROOT_DIR / "config"

KEYPOINTS_CSV = RAW_DATA_DIR / "keypoints.csv"
MODEL_PATH = MODELS_DIR / "gesture_mlp.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
LABELS_JSON = CONFIG_DIR / "labels.json"
HAND_LANDMARKER_TASK = MODELS_DIR / "hand_landmarker.task"


def get_supported_labels() -> list[str]:
    with LABELS_JSON.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    labels = payload.get("labels", [])
    return [str(x).strip().upper() for x in labels if str(x).strip()]
