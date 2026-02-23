# KNN Classification Model

K-Nearest Neighbors (KNN) classifier implemented in Python with full preprocessing and evaluation.

## Dataset

- **Index levels** from `archive.zip` (file `Index_Levels.csv` in this repo).
  - **Features**: Open, High, Low, Close, Volume (daily OHLCV for each index).
  - **Target**: Index name (10 classes) — DAX40, FTSE100, SP500, BIST100, BOVESPA, NIKKEI225, IDX, SSE, TADAWUL, NIFTY50.
  - The script reads the CSV from inside the zip; keep `archive.zip` in the same folder as `knn_classification.py`.

## Steps

1. **Load data** — Read `Index_Levels.csv` from `archive.zip`; encode target (Index) with `LabelEncoder`.
2. **Preprocessing**
   - Remove missing values and duplicates
   - Optional: remove outliers (3-sigma rule on feature columns only)
   - Normalize numeric features with `StandardScaler`
   - Target (Index) is encoded in the load step with `LabelEncoder`
   - Train/test split (80/20, `random_state=42`)
3. **Training** — Fit `KNeighborsClassifier` with `n_neighbors=5`.
4. **Evaluation** — Accuracy, precision, recall, F1 (macro), and confusion matrix (saved as `confusion_matrix.png`).

## Two implementations

| File | Description |
|------|--------------|
| `knn_classification.py` | Full pipeline using **sklearn** (StandardScaler, train_test_split, KNeighborsClassifier, metrics). |
| `knn_from_scratch.py` | Same pipeline **without sklearn**: manual scaling, split, KNN (Euclidean distance + majority vote), and metrics. Saves `confusion_matrix_from_scratch.png`. |

## Setup & Run

```bash
pip install -r requirements.txt
```

**With sklearn:**
```bash
python knn_classification.py
```

**From scratch (no sklearn):**
```bash
python knn_from_scratch.py
```

## Results

After running the script you will see:

- **Metrics**: Accuracy, Precision (macro), Recall (macro), F1 Score (macro)
- **Confusion matrix**: Plot displayed and saved as `confusion_matrix.png`

Example output (Index dataset, after preprocessing):

| Metric    | Value   |
|----------|---------|
| Accuracy | (see run) |
| Precision| (macro)  |
| Recall   | (macro)  |
| F1 Score | (macro)  |

Confusion matrix shows per-class predictions (rows = true, columns = predicted); axis labels are the index names (e.g. SP500, DAX40).

## Push to GitHub

```bash
git init
git add .
git commit -m "KNN classification model"
git branch -M main
git remote add origin https://github.com/<your-username>/knn-project.git
git push -u origin main
```

Replace `<your-username>` with your GitHub username.
