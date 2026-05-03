# Credit-Card Default Prediction Service

**Датасет:** [Default of Credit Card Clients, UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)  
**Docker Hub:** [`dyakowv/credit-default-api`](https://hub.docker.com/r/dyakowv/credit-default-api)

ML-сервис прогнозирования дефолта по кредитным картам — полный цикл от обучения модели до контейнеризации и A/B-тестирования.

---

## Структура репозитория

```
.
├── app/
│   ├── api.py              # Flask-приложение (/predict, /health)
│   └── model_handler.py    # Загрузка модели и инференс
├── models/
│   ├── train_model.py      # Скрипт обучения
│   ├── model_v1.pkl        # LogisticRegression (baseline)
│   └── model_v2.pkl        # RandomForest (challenger)
├── tests/
│   └── test_api.py         # 12 pytest-тестов
├── docker/
│   └── Dockerfile
├── data/
│   └── UCI_Credit_Card.csv
├── data_for_training.csv
├── .dockerignore
├── docker-compose.yml
├── requirements.txt
├── ab_test_plan.md
└── README.md
```

---

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m models.train_model   # создать model_v1.pkl, model_v2.pkl
python -m app.api              # запустить на http://localhost:8080
```

---

## API

### GET /health

```bash
curl http://localhost:8080/health
# {"available_versions":["v1","v2"],"status":"healthy"}
```

### POST /predict

Параметр `?version=v1` (по умолчанию) или `?version=v2`.

**Тело запроса:** 23 числовых поля датасета UCI.

| Поле | Тип | Описание |
|---|---|---|
| `LIMIT_BAL` | float | Кредитный лимит (TWD) |
| `SEX` | int | 1 = муж, 2 = жен |
| `EDUCATION` | int | 1–4 |
| `MARRIAGE` | int | 1–3 |
| `AGE` | int | Возраст |
| `PAY_0`…`PAY_6` | int | Статус погашения (−2..8) |
| `BILL_AMT1`…`BILL_AMT6` | float | Сумма счёта (TWD) |
| `PAY_AMT1`…`PAY_AMT6` | float | Сумма платежа (TWD) |

```bash
curl -s -X POST "http://localhost:8080/predict?version=v1" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,
       "PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,
       "BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,
       "BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,
       "PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,
       "PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0}'
# {"model_version":"model_v1","prediction":1,"probability":0.775}
```

**Ответ:** `prediction` (0/1), `probability` ([0,1]), `model_version`.

**Коды ошибок:** `400` — нет признаков или неизвестная версия; `500` — ошибка инференса.

---

## Модели

| Версия | Алгоритм | ROC-AUC |
|---|---|---|
| v1 | LogisticRegression + StandardScaler | 0.708 |
| v2 | RandomForestClassifier (200 деревьев) | 0.757 |

Модели сериализованы через `joblib`. Переобучить: `python -m models.train_model`

---

## Тесты

```bash
.venv/bin/python -m pytest tests/ -v
```

---

## Docker

**Pull с Docker Hub:**
```bash
docker pull dyakowv/credit-default-api:latest
docker run -p 8080:8080 dyakowv/credit-default-api:latest
```

**Сборка из исходников:**
```bash
docker build -f docker/Dockerfile -t credit-default-api .
docker run -p 8080:8080 credit-default-api
```

**Docker Compose:**
```bash
docker-compose up -d
curl http://localhost:8080/health
docker-compose down
```

---

## uWSGI + NGINX

Встроенный Flask-сервер однопоточный и не предназначен для production. Стандартная схема:

```
Client → NGINX (reverse proxy, SSL) → uWSGI (пул воркеров) → Flask
```

Nginx работает как фронтенд-сервер (обратный прокси), обраьатывя HTTP-запросы и статику, а uWSGI — как бэкенд-сервер приложений, выполняя Python-код. Для учебного проекта достаточно `gunicorn`.

---

## ONNX-ML

Конвертация в аппаратно-независимый формат (инференс без scikit-learn):

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

pipeline = joblib.load("models/model_v1.pkl")
onnx_model = convert_sklearn(pipeline,
    initial_types=[("float_input", FloatTensorType([None, 23]))])
with open("models/model_v1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

---

## Архитектура: монолит vs микросервисы

Выбран **монолит**: API, модель и предобработка в одном образе.

| Критерий | Монолит (выбран) | Микросервисы |
|---|---|---|
| Деплой | Один контейнер | Оркестрация, service discovery |
| Латентность | Вызов внутри процесса | Сетевые вызовы |
| Масштабирование | Реплики контейнера | Независимо по компонентам |
| Обоснование | Один домен, одна модель | Нужно при 5+ компонентах |

---

## Брокер сообщений (концепт)

При высокой нагрузке `/predict` переходит на асинхронную обработку через **RabbitMQ**:

```
API (producer) → queue → Worker (consumer) → БД / callback
```

Сценарии: батч-скоринг ночью, логирование предсказаний для A/B-анализа, триггер переобучения.

---

## Логирование

Каждый запрос логируется в JSON (stdout контейнера):

```json
{"time": "...", "level": "INFO",
 "message": {"endpoint": "/predict", "version": "v1", "prediction": 1, "probability": 0.775}}
```

В production сбор логов через **ELK** (Filebeat → Logstash → Elasticsearch → Kibana) или **Grafana + Loki**.

---

## MLOps: DVC и MLflow

**DVC** — версионирование данных. `data_for_training.csv` хранится в S3/GCS, в Git коммитится только `.dvc`-указатель.

```bash
dvc init && dvc add data_for_training.csv && dvc push
```

**MLflow** — отслеживание экспериментов. При обучении логируются параметры, метрики и артефакт модели; `mlflow ui` показывает сравнительную таблицу запусков.

```python
with mlflow.start_run():
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("roc_auc", 0.757)
    mlflow.sklearn.log_model(pipeline_v2, "model_v2")
```

---

## Бизнес-метрики

**Expected Loss** — суммарные потери по пропущенным дефолтам (FN):
```
Expected Loss = P(default) × LIMIT_BAL × LGD   (LGD ≈ 0.45)
```

**Approval Rate** — доля одобренных заявок при фиксированной доле дефолтов ≤ 10%. v2 с более высоким ROC-AUC одобрит больше хороших клиентов при том же уровне риска.
