
FROM python:3.13-slim

# не дает Python генерировать .pyc файлы и включает неблокирующий вывод в stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Устанавливаем зависимости сначала (кэшируется отдельно от кода)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения и предобученные модели
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8080

# Запуск в виде модуля, чтобы абсолютные импорты пакетов (app.*) работали корректно
CMD ["python", "-m", "app.api"]
