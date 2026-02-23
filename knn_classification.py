"""
K-Nearest Neighbors (KNN) Classification Model
Dataset: Index_Levels.csv from archive.zip (stock indices by country).
Target: which index (e.g. SP500, DAX40, NIFTY50). Features: Open, High, Low, Close, Volume.
"""

import zipfile
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Path to the dataset zip (same folder as script)
ZIP_PATH = Path(__file__).resolve().parent / "archive.zip"
CSV_NAME = "Index_Levels.csv"
TARGET_COL = "Index"
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ========== 1. Download / Load Data ==========
def load_data(zip_path=ZIP_PATH, csv_name=CSV_NAME):
    """Load dataset from archive.zip (Index_Levels.csv)."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    # Encode categorical target (Index -> 0, 1, ...)
    le = LabelEncoder()
    df["target"] = le.fit_transform(df[TARGET_COL].astype(str))

    # Use only numeric features (drop Date and original Index)
    use_cols = FEATURE_COLS + ["target"]
    df = df[[c for c in use_cols if c in df.columns]]

    print("Dataset shape:", df.shape)
    print("Target classes:", le.classes_.tolist())
    print(df.head())
    return df, le


# ========== 2. Data Preprocessing ==========
def preprocess(df):
    """Remove missing values, duplicates, optional outliers; scale features."""
    # a. Remove missing values and duplicates
    df = df.dropna()
    df = df.drop_duplicates()

    # b. Handle outliers (optional) — apply only to numeric feature columns
    feature_cols = [c for c in df.columns if c != "target"]
    if feature_cols:
        mask = (
            np.abs(df[feature_cols] - df[feature_cols].mean())
            <= (3 * df[feature_cols].std())
        ).all(axis=1)
        df = df[mask]

    # c. Normalize / scale numeric values
    X = df.drop("target", axis=1)
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # d. Categorical encoding is done in load_data() for this dataset.

    # e. Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler


# ========== 3. Implement KNN Model ==========
def train_knn(X_train, y_train, n_neighbors=5):
    """Train KNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


# ========== 4. Evaluate Model ==========
def evaluate(knn, X_test, y_test, class_names=None, plot_cm=True):
    """Compute metrics and optionally plot confusion matrix."""
    y_pred = knn.predict(X_test)

    print("\n--- Evaluation Metrics ---")
    print("Accuracy:  ", accuracy_score(y_test, y_pred))
    print(
        "Precision: ",
        precision_score(y_test, y_pred, average="macro", zero_division=0),
    )
    print(
        "Recall:    ",
        recall_score(y_test, y_pred, average="macro", zero_division=0),
    )
    print(
        "F1 Score:  ",
        f1_score(y_test, y_pred, average="macro", zero_division=0),
    )

    cm = confusion_matrix(y_test, y_pred)
    if plot_cm:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names if class_names is not None else None,
        )
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix (KNN)")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        plt.show()

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": cm,
    }


# ========== Main ==========
def main():
    df, label_encoder = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess(df)
    knn = train_knn(X_train, y_train, n_neighbors=5)
    metrics = evaluate(
        knn,
        X_test,
        y_test,
        class_names=label_encoder.classes_.tolist(),
        plot_cm=True,
    )
    return metrics


if __name__ == "__main__":
    main()
