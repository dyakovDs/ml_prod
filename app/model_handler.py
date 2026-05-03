# загрузка и инференс моделей

from pathlib import Path

import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


class ModelHandler:

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.version = self.model_path.stem          # "model_v1" → version label
        self._pipeline = joblib.load(self.model_path)

    def predict(self, features: dict) -> dict:
        """
        Run inference on a single sample.
        Запуск инференса на одном сэмпле

        Параметры
        ----------
        features : dict
            Маппинг feature name → value для каждой колонки в FEATURE_COLUMNS.

        Возвращает
        -------
        dict with keys: prediction (int), probability (float), model_version (str)
        """
        df = pd.DataFrame([features])[FEATURE_COLUMNS]
        prediction = int(self._pipeline.predict(df)[0])
        probability = float(self._pipeline.predict_proba(df)[0][1])
        return {
            "prediction": prediction,
            "probability": round(probability, 6),
            "model_version": self.version,
        }
