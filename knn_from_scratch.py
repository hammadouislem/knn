"""
K-Nearest Neighbors (KNN) Classification — implemented from scratch (no sklearn).
Uses same dataset as knn_classification.py: Index_Levels.csv from archive.zip.
"""

import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ZIP_PATH = Path(__file__).resolve().parent / "archive.zip"
CSV_NAME = "Index_Levels.csv"
TARGET_COL = "Index"
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ========== 1. Load Data (no sklearn) ==========
def load_data(zip_path=ZIP_PATH, csv_name=CSV_NAME):
    """Load dataset from archive.zip; encode target manually."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    # Encode target without sklearn: map class names -> 0, 1, ...
    unique_classes = df[TARGET_COL].astype(str).unique()
    class_to_idx = {c: i for i, c in enumerate(sorted(unique_classes))}
    df["target"] = df[TARGET_COL].astype(str).map(class_to_idx)

    use_cols = FEATURE_COLS + ["target"]
    df = df[[c for c in use_cols if c in df.columns]]

    idx_to_class = {i: c for c, i in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    print("Dataset shape:", df.shape)
    print("Target classes:", class_names)
    print(df.head())
    return df, class_names


# ========== 2. Preprocessing (no sklearn) ==========
def standard_scale(X, mean=None, std=None):
    """Z-score scaling: (X - mean) / std. If mean/std None, compute from X."""
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        std[std == 0] = 1.0  # avoid division by zero
    return (X - mean) / std, mean, std


def train_test_split_np(X, y, test_size=0.2, random_state=42):
    """Random 80/20 train/test split."""
    n = len(X)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def preprocess(df):
    """Dropna, drop_duplicates, optional outliers, scale, split."""
    df = df.dropna().drop_duplicates()

    feature_cols = [c for c in df.columns if c != "target"]
    if feature_cols:
        mean_f = df[feature_cols].mean()
        std_f = df[feature_cols].std()
        mask = (
            np.abs(df[feature_cols] - mean_f) <= (3 * std_f)
        ).all(axis=1)
        df = df[mask]

    X = df.drop("target", axis=1).values.astype(np.float64)
    y = df["target"].values.astype(np.intp)

    X_scaled, mean, std = standard_scale(X)
    X_train, X_test, y_train, y_test = train_test_split_np(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, (mean, std)


# ========== 3. KNN from scratch ==========
def euclidean_distance(X, X_train):
    """Compute pairwise Euclidean distances: (n_test, n_train)."""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    X_sq = (X ** 2).sum(axis=1, keepdims=True)
    X_train_sq = (X_train ** 2).sum(axis=1)
    cross = X @ X_train.T
    dist_sq = np.maximum(X_sq + X_train_sq - 2 * cross, 0)
    return np.sqrt(dist_sq)


class KNeighborsClassifierScratch:
    """KNN classifier: store training data, predict by k-nearest majority vote.
    Uses batched distance computation to avoid large memory allocation."""

    def __init__(self, n_neighbors=5, batch_size=512):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size  # process test samples in batches to save memory
        self.X_train = None
        self.y_train = None
        self.n_classes = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.intp)
        self.n_classes = len(np.unique(y))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_test = X.shape[0]
        k = min(self.n_neighbors, self.X_train.shape[0])
        pred = np.empty(n_test, dtype=np.intp)

        for start in range(0, n_test, self.batch_size):
            end = min(start + self.batch_size, n_test)
            X_batch = X[start:end]
            dist = euclidean_distance(X_batch, self.X_train)  # (batch, n_train)
            nearest = np.argpartition(dist, k - 1, axis=1)[:, :k]
            neighbor_labels = self.y_train[nearest]
            for i in range(end - start):
                counts = np.bincount(
                    neighbor_labels[i], minlength=self.n_classes
                )
                pred[start + i] = np.argmax(counts)
        return pred


# ========== 4. Metrics from scratch ==========
def accuracy_score_np(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix_np(y_true, y_pred, n_classes=None):
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.intp)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_from_cm(cm, zero_division=0):
    """Macro precision, recall, F1 from confusion matrix."""
    n = cm.shape[0]
    precisions = np.zeros(n)
    recalls = np.zeros(n)
    for i in range(n):
        col = cm[:, i].sum()
        row = cm[i, :].sum()
        precisions[i] = cm[i, i] / col if col else zero_division
        recalls[i] = cm[i, i] / row if row else zero_division
    f1s = np.zeros(n)
    for i in range(n):
        p, r = precisions[i], recalls[i]
        f1s[i] = 2 * p * r / (p + r) if (p + r) > 0 else zero_division
    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
    )


def evaluate(knn, X_test, y_test, class_names=None, plot_cm=True):
    """Compute metrics and plot confusion matrix (matplotlib only)."""
    y_pred = knn.predict(X_test)

    acc = accuracy_score_np(y_test, y_pred)
    cm = confusion_matrix_np(y_test, y_pred, n_classes=knn.n_classes)
    precision, recall, f1 = precision_recall_f1_from_cm(cm)

    print("\n--- Evaluation Metrics (from scratch) ---")
    print("Accuracy:  ", acc)
    print("Precision: ", precision)
    print("Recall:    ", recall)
    print("F1 Score:  ", f1)

    if plot_cm:
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        if class_names is not None:
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (KNN from scratch)")
        plt.tight_layout()
        plt.savefig("confusion_matrix_from_scratch.png", dpi=150)
        plt.show()

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
    }


# ========== Main ==========
def main():
    df, class_names = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess(df)
    knn = KNeighborsClassifierScratch(n_neighbors=5)
    knn.fit(X_train, y_train)
    metrics = evaluate(
        knn, X_test, y_test, class_names=class_names, plot_cm=True
    )
    return metrics


if __name__ == "__main__":
    main()
