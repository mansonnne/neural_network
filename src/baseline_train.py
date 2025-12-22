"""
Baseline pipeline:
- Load data (full or sample)
- Drop columns with too many NaNs
- Split train/val/test with stratification
- Preprocess: median for numeric, "unknown" for cats, one-hot, scale nums
- Model: LogisticRegression (class_weight='balanced')
- Metrics: accuracy, precision, recall, f1, roc_auc, confusion matrix
"""

import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/data.csv"
TARGET_COL = "target"
DEFAULT_NROWS = 5000  # быстрый прогон по умолчанию


def load_data(path: str, nrows: int | None) -> pd.DataFrame:
    print(f"Читаю {'весь датасет' if nrows is None else f'сэмпл {nrows} строк'} из {path}...")
    return pd.read_csv(path, nrows=nrows)


def split_data(
    df: pd.DataFrame, target: str, test_size: float, val_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]

    # сначала отделяем тест
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # потом отделяем валидацию от train_val
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, stratify=y_train_val, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocess_and_model(
    X: pd.DataFrame, missing_thresh: float, max_iter: int
) -> Pipeline:
    # Выделяем типы
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Дроп признаков с высокой долей пропусков
    missing_ratio = X.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > missing_thresh].index.tolist()
    if drop_cols:
        print(f"Дропаем колонки с пропусками > {missing_thresh:.2f}: {drop_cols}")
        cat_cols = [c for c in cat_cols if c not in drop_cols]
        num_cols = [c for c in num_cols if c not in drop_cols]

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", fill_value="unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        n_jobs=1,  # избегаем параллели, чтобы не упереться в ограничения среды
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return clf


def evaluate(model: Pipeline, X, y, split_name: str):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    roc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds)
    print(f"\n[{split_name}]")
    print(f"accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} roc_auc={roc:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "cm": cm.tolist(),
    }


def main(args: argparse.Namespace):
    df = load_data(args.data_path, None if args.full else args.nrows)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target=TARGET_COL,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )

    clf = build_preprocess_and_model(
        X_train,
        missing_thresh=args.missing_thresh,
        max_iter=args.max_iter,
    )

    print("\nОбучаю модель...")
    clf.fit(X_train, y_train)

    print("\nОценка:")
    metrics = {}
    metrics["train"] = evaluate(clf, X_train, y_train, "train")
    metrics["val"] = evaluate(clf, X_val, y_val, "val")
    metrics["test"] = evaluate(clf, X_test, y_test, "test")

    print("\nГотово.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline: предобработка + логрег")
    parser.add_argument("--data-path", default=DATA_PATH, help="Путь к CSV с данными")
    parser.add_argument("--full", action="store_true", help="Читать весь датасет (по умолчанию сэмпл)")
    parser.add_argument("--nrows", type=int, default=DEFAULT_NROWS, help="Сэмпл строк, если не full")
    parser.add_argument("--test-size", type=float, default=0.2, help="Доля test")
    parser.add_argument("--val-size", type=float, default=0.2, help="Доля val (от всего датасета)")
    parser.add_argument("--missing-thresh", type=float, default=0.65, help="Порог доли пропусков для дропа колонок")
    parser.add_argument("--max-iter", type=int, default=500, help="max_iter для логистической регрессии")
    parser.add_argument("--seed", type=int, default=42, help="random_state")
    args = parser.parse_args()
    main(args)

