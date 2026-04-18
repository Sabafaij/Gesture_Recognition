from __future__ import annotations
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from config import KEYPOINTS_CSV, MODEL_PATH, LABEL_ENCODER_PATH, MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP model on gesture landmarks.")
    parser.add_argument("--test-size", default=0.2, type=float, help="Validation split ratio")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not KEYPOINTS_CSV.exists():
        raise FileNotFoundError(f"Training file not found: {KEYPOINTS_CSV}")

    df = pd.read_csv(KEYPOINTS_CSV)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    y = df["label"].astype(str).str.upper().values
    X = df.drop(columns=["label"]).values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=args.test_size, random_state=args.seed, stratify=y_encoded
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, LABEL_ENCODER_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
