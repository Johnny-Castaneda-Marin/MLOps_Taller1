import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Penguin Classifier API")

df_metrics = pd.read_pickle("report/model_metrics.pkl")

metrics_by_model = {
    row["model"].lower(): {
        "train_accuracy": row["train_accuracy"],
        "test_accuracy": row["test_accuracy"],
        "test_precision": row["test_precision"],
        "test_recall": row["test_recall"],
        "test_f1": row["test_f1"],
    }
    for _, row in df_metrics.iterrows()
}

MODELS = {
    "randomforest": {
        "model": "Random Forest Classifier",
        "endpoint": "POST /classify/randomforest",
        "metrics": metrics_by_model.get("randomforest", {}),
    },
    "svm": {
        "model": "Support Vector Classifier",
        "endpoint": "POST /classify/svm",
        "metrics": metrics_by_model.get("svm", {}),
    },
    "gradientboosting": {
        "model": "Gradient Boosting Classifier",
        "endpoint": "POST /classify/gradientboosting",
        "metrics": metrics_by_model.get("gradientboosting", {}),
    },
}

model_instances = {
    "randomforest": joblib.load("modelos/randomforest_model.pkl"),
    "svm": joblib.load("modelos/svm_model.pkl"),
    "gradientboosting": joblib.load("modelos/gradientboosting_model.pkl"),
}

scaler = joblib.load("modelos/scaler.pkl")

SPECIES_MAP = {1: "Adelie", 2: "Chinstrap", 3: "Gentoo"}


class PenguinInput(BaseModel):
    island: int = Field(default=1, examples=[1])
    bill_length_mm: float = Field(default=39.1, examples=[39.1])
    bill_depth_mm: float = Field(default=18.7, examples=[18.7])
    flipper_length_mm: int = Field(default=181, examples=[181])
    body_mass_g: int = Field(default=3750, examples=[3750])
    sex: int = Field(default=1, examples=[1])
    year: int = Field(default=2007, examples=[2007])

    @field_validator("island")
    def validate_island(cls, v):
        if v not in (1, 2, 3):
            raise ValueError("island debe ser 1, 2 o 3")
        return v

    @field_validator("bill_length_mm")
    def validate_bill_length(cls, v):
        if not (10.0 <= v <= 100.0):
            raise ValueError("bill_length_mm debe estar entre 10.0 y 100.0")
        return v

    @field_validator("bill_depth_mm")
    def validate_bill_depth(cls, v):
        if not (5.0 <= v <= 35.0):
            raise ValueError("bill_depth_mm debe estar entre 5.0 y 35.0")
        return v

    @field_validator("flipper_length_mm")
    def validate_flipper_length(cls, v):
        if not (100 <= v <= 300):
            raise ValueError("flipper_length_mm debe estar entre 100 y 300")
        return v

    @field_validator("body_mass_g")
    def validate_body_mass(cls, v):
        if not (1000 <= v <= 10000):
            raise ValueError("body_mass_g debe estar entre 1000 y 10000")
        return v

    @field_validator("sex")
    def validate_sex(cls, v):
        if v not in (0, 1):
            raise ValueError("sex debe ser 0 o 1")
        return v

    @field_validator("year")
    def validate_year(cls, v):
        if not (2000 <= v <= 2030):
            raise ValueError("year debe estar entre 2000 y 2030")
        return v


def _build_features(data: PenguinInput) -> np.ndarray:
    bill_ratio = data.bill_length_mm / data.bill_depth_mm
    body_mass_kg = data.body_mass_g / 1000
    return np.array([[
        data.island, data.bill_length_mm, data.bill_depth_mm,
        data.flipper_length_mm, data.body_mass_g, data.sex,
        data.year, bill_ratio, body_mass_kg
    ]])


@app.get("/models")
async def list_models():
    return {
        "available_models": [
            {
                "name": name,
                "model": info["model"],
                "metrics": info["metrics"],
                "endpoint": info["endpoint"],
            }
            for name, info in MODELS.items()
        ]
    }


@app.post("/classify/{model_name}")
async def classify(model_name: str, data: PenguinInput):
    if model_name not in model_instances:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{model_name}' no encontrado. Usa GET /models para ver los disponibles.",
        )
    try:
        features = _build_features(data)
        scaled = scaler.transform(features)
        prediction = int(model_instances[model_name].predict(scaled)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    species_name = SPECIES_MAP.get(prediction)
    if species_name is None:
        raise HTTPException(status_code=404, detail="Especie no encontrada")
    return {
        "model": model_name,
        "species_id": prediction,
        "species_name": species_name,
    }
