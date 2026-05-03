"""
Flask API для кредитного скоринга.

Эндпоинты
---------
GET  /health        — проверка работоспособности
POST /predict       — предсказание на одном сэмпле

Параметры запроса
---------------
?version=v1  (default)  — используем LogisticRegression
?version=v2             — используем RandomForest 

Пример тела запроса (JSON)
-------------------
{
  "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
  "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
  "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
  "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
  "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
  "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
}

Пример ответа (JSON)
---------------
{ "prediction": 1, "probability": 0.734, "model_version": "model_v1" }
"""

import json
import logging
from pathlib import Path

from flask import Flask, jsonify, request

from app.model_handler import FEATURE_COLUMNS, ModelHandler

# логирование
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}',
)
logger = logging.getLogger(__name__)

# приложение и модели
app = Flask(__name__)

_models_dir = Path(__file__).parent.parent / "models"
_models: dict[str, ModelHandler] = {
    "v1": ModelHandler(_models_dir / "model_v1.pkl"),
    "v2": ModelHandler(_models_dir / "model_v2.pkl"),
}


# ручки
@app.get("/health")
def health():
    # вернет 200 если сервис работает и модели загружены
    return jsonify({"status": "healthy", "available_versions": list(_models.keys())}), 200


@app.post("/predict")
def predict():
    """
    Эндопинт возвращает предсказание по одному сэмплу.

    Принимает JSON с 23 фичами.
    Используем ?version=v1 (default) или ?version=v2 для выбора модели.
    """
    data: dict = request.get_json(force=True, silent=True) or {}

    # Валидация входных данных
    missing = [col for col in FEATURE_COLUMNS if col not in data]
    if missing:
        return jsonify({"error": "missing_features", "details": missing}), 400

    version = request.args.get("version", "v1")
    if version not in _models:
        return jsonify({"error": "unknown_version", "available": list(_models.keys())}), 400

    # Извелкаем нужные колонки
    features = {col: data[col] for col in FEATURE_COLUMNS}

    try:
        result = _models[version].predict(features)
    except Exception as exc:
        logger.error(json.dumps({"endpoint": "/predict", "error": str(exc)}))
        return jsonify({"error": "prediction_failed", "details": str(exc)}), 500

    logger.info(json.dumps({
        "endpoint": "/predict",
        "version": version,
        "prediction": result["prediction"],
        "probability": result["probability"],
    }))
    return jsonify(result), 200


# входной эндпоинт
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
