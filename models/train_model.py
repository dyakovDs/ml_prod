"""
Скрипт для обучения моделей на подготовленных данных.

Обучаем две модели:
    models/model_v1.pkl  — LogisticRegression (scaled, class-balanced)
    models/model_v2.pkl  — RandomForestClassifier (class-balanced)

Запуск из корневой папки проекта:
    python -m models.train_model
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data_for_training.csv"
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET_COLUMN = "default.payment.next.month"


def evaluate(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["no default", "default"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")


def train_and_save() -> None:
    print(f"Загрузка данных из {DATA_PATH} …")
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Трен. данные: {len(X_train)} строк  |  Тест. данные: {len(X_test)} строк")
    print(f"Доля дефолта — трен. выборка: {y_train.mean():.3f}  тест. выборка: {y_test.mean():.3f}")

    # ---------- v1: Logistic Regression ----------
    pipeline_v1 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )),
    ])
    print("\nОбучение v1 (LogisticRegression) …")
    pipeline_v1.fit(X_train, y_train)
    evaluate("Модель v1 — LogisticRegression", pipeline_v1, X_test, y_test)
    path_v1 = MODELS_DIR / "model_v1.pkl"
    joblib.dump(pipeline_v1, path_v1)
    print(f"Сохранено → {path_v1}")

    # ---------- v2: Random Forest ----------
    pipeline_v2 = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])
    print("\nОбучение v2 (RandomForestClassifier) …")
    pipeline_v2.fit(X_train, y_train)
    evaluate("Модель v2 — RandomForestClassifier", pipeline_v2, X_test, y_test)
    path_v2 = MODELS_DIR / "model_v2.pkl"
    joblib.dump(pipeline_v2, path_v2)
    print(f"Сохранено → {path_v2}")


if __name__ == "__main__":
    train_and_save()
