# Penguin Classifier API

API de clasificación de especies de pingüinos con 3 modelos de Machine Learning, construida con FastAPI y desplegada en Docker.

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Entrenamiento de los Modelos](#entrenamiento-de-los-modelos)
- [Desarrollo de la API](#desarrollo-de-la-api)
- [Contenerización con Docker](#contenerización-con-docker)
- [Pruebas y Resultados](#pruebas-y-resultados)

---

## Descripción General

Pipeline MLOps que cubre desde el entrenamiento de 3 modelos de clasificación hasta su despliegue como API REST en un contenedor Docker. Los modelos clasifican pingüinos en 3 especies a partir de medidas morfológicas.

| ID | Especie   |
|----|-----------|
| 1  | Adelie    |
| 2  | Chinstrap |
| 3  | Gentoo    |

---

## Estructura del Proyecto

```
├── API/
│   ├── app.py                          # API FastAPI
│   ├── modelos/
│   │   ├── randomforest_model.pkl      # Modelo Random Forest
│   │   ├── svm_model.pkl              # Modelo SVM
│   │   ├── gradientboosting_model.pkl # Modelo Gradient Boosting
│   │   └── scaler.pkl                 # StandardScaler
│   └── report/
│       └── model_metrics.pkl          # DataFrame con métricas de los modelos
├── Docker/
│   └── Dockerfile
├── images/                            # Capturas de pantalla
├── model_train_and_save/
│   ├── train.ipynb                    # Notebook de entrenamiento
│   ├── penguins_v1.csv               # Dataset
│   └── requirements.txt              # Dependencias Python
└── README.md
```

---

## Entrenamiento de los Modelos

El notebook `model_train_and_save/train.ipynb` ejecuta el siguiente pipeline:

1. Carga del dataset `penguins_v1.csv` (333 registros, 9 columnas)
2. Limpieza: verificación de nulos y duplicados
3. Transformación: separación de features y variable objetivo (`species`)
4. Validación: estadísticas descriptivas y distribución de clases
5. Feature engineering: creación de `bill_ratio` y `body_mass_kg`
6. Split train/test (80/20, estratificado)
7. Escalado con `StandardScaler`
8. Entrenamiento de 3 modelos:
   - **Random Forest** (n_estimators=100, max_depth=10)
   - **SVM** (kernel=rbf, C=1.0)
   - **Gradient Boosting** (n_estimators=100, max_depth=5, lr=0.1)
9. Evaluación con accuracy, precision, recall y f1-score
10. Serialización de modelos en `API/modelos/`, scaler y DataFrame de métricas en `API/report/`

### Features de entrada

| Feature            | Tipo  | Descripción                          |
|--------------------|-------|--------------------------------------|
| `island`           | int   | Isla (1, 2 o 3)                     |
| `bill_length_mm`   | float | Largo del pico en mm                 |
| `bill_depth_mm`    | float | Profundidad del pico en mm           |
| `flipper_length_mm`| int   | Largo de la aleta en mm              |
| `body_mass_g`      | int   | Masa corporal en gramos              |
| `sex`              | int   | Sexo (0: female, 1: male)            |
| `year`             | int   | Año de observación                   |
| `bill_ratio`       | float | Ratio largo/profundidad (calculado)  |
| `body_mass_kg`     | float | Masa en kg (calculado)               |

---

## Desarrollo de la API

La API fue construida con **FastAPI** y **Pydantic v2**. Al iniciar, carga los 3 modelos serializados, el scaler y el DataFrame de métricas desde disco.

### Endpoints

#### `GET /models`

Retorna la lista de modelos disponibles con sus métricas de evaluación.

```json
{
  "available_models": [
    {
      "name": "randomforest",
      "model": "Random Forest Classifier",
      "metrics": {
        "train_accuracy": 1.0,
        "test_accuracy": 0.985,
        "test_precision": 0.986,
        "test_recall": 0.985,
        "test_f1": 0.985
      },
      "endpoint": "POST /classify/randomforest"
    }
  ]
}
```

#### `POST /classify/{model_name}`

Recibe las características de un pingüino y retorna la especie predicha usando el modelo indicado.

Request:
```json
{
  "island": 1,
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750,
  "sex": 1,
  "year": 2007
}
```

Response:
```json
{
  "model": "svm",
  "species_id": 1,
  "species_name": "Adelie"
}
```

### Validaciones

Todos los campos de entrada se validan con `field_validator` de Pydantic:

| Campo              | Regla                    |
|--------------------|--------------------------|
| `island`           | 1, 2 o 3                |
| `bill_length_mm`   | entre 10.0 y 100.0      |
| `bill_depth_mm`    | entre 5.0 y 35.0        |
| `flipper_length_mm`| entre 100 y 300          |
| `body_mass_g`      | entre 1000 y 10000       |
| `sex`              | 0 o 1                   |
| `year`             | entre 2000 y 2030        |

Si un campo no cumple, la API retorna `422 Unprocessable Entity` con el detalle del error.

### Documentación interactiva

FastAPI genera Swagger UI automáticamente con valores de ejemplo prellenados:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Contenerización con Docker

### Dockerfile

El Dockerfile está en `Docker/Dockerfile` y usa como contexto de build la raíz del proyecto:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY model_train_and_save/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY API/app.py .
COPY API/modelos/ modelos/
COPY API/report/ report/
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Copia `app.py`, la carpeta `modelos/` (3 modelos + scaler) y `report/` (DataFrame de métricas) manteniendo las rutas relativas que espera la API.

### Construcción

```bash
docker build -f Docker/Dockerfile -t penguin-api .
```

![Docker Build](images/docker_build.png)

### Ejecución

```bash
docker run -d --name penguin-api -p 8000:8000 penguin-api
```

![Docker Run](images/docker_run.png)

---

## Pruebas y Resultados

### Métricas de los modelos

| Modelo            | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1  |
|-------------------|---------------|---------------|----------------|-------------|----------|
| Random Forest     | 1.0000        | 0.9851        | 0.9856         | 0.9851      | 0.9849   |
| SVM               | 1.0000        | 1.0000        | 1.0000         | 1.0000      | 1.0000   |
| Gradient Boosting | 1.0000        | 0.9851        | 0.9856         | 0.9851      | 0.9849   |

Los 3 modelos logran accuracy perfecta en entrenamiento. En test, SVM alcanza 100% en todas las métricas, mientras que Random Forest y Gradient Boosting comparten un test accuracy de 98.51%.

Las pruebas se realizaron usando Postman contra la API corriendo en `http://localhost:8000`.

### Consulta de modelos disponibles

`GET http://localhost:8000/models`

![Postman /models](images/postman_models.png)

### Clasificación con Random Forest

`POST http://localhost:8000/classify/randomforest`

Body (JSON):
```json
{
  "island": 1,
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750,
  "sex": 1,
  "year": 2007
}
```

![Postman Random Forest](images/postman_classify_randomforest.png)

### Clasificación con SVM

`POST http://localhost:8000/classify/svm`

Body (JSON):
```json
{
  "island": 1,
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750,
  "sex": 1,
  "year": 2007
}
```

![Postman SVM](images/postman_classify_svm.png)

### Clasificación con Gradient Boosting

`POST http://localhost:8000/classify/gradientboosting`

Body (JSON):
```json
{
  "island": 1,
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750,
  "sex": 1,
  "year": 2007
}
```

![Postman Gradient Boosting](images/postman_classify_gradientboosting.png)

### Validación de errores

`POST http://localhost:8000/classify/svm`

Body (JSON) con valor inválido en `island`:
```json
{
  "island": 5,
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750,
  "sex": 1,
  "year": 2007
}
```

![Postman validación error](images/postman_validation_error.png)

### Modelo inexistente

`POST http://localhost:8000/classify/xgboost`

![Postman modelo inexistente](images/postman_model_not_found.png)
