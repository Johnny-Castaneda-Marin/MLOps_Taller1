FROM python:3.11-slim

WORKDIR /app

COPY model_train_and_save/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY API/app.py .
COPY API/modelos/ modelos/
COPY API/report/ report/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
