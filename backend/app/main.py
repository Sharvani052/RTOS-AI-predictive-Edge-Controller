from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ==========================================
# APPLICATION SETUP
# ==========================================

app = FastAPI(
    title="Edge AI Predictive Maintenance API",
    description="AI-powered predictive maintenance controller for Industry 5.0",
    version="1.0.0"
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "predictive_maintenance_model.joblib"

model = joblib.load(MODEL_PATH)


# ==========================================
# MODEL FEATURES
# ==========================================

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# ==========================================
# INPUT DATA MODEL
# ==========================================

class MachineData(BaseModel):
    air_temperature: float = Field(
        ...,
        description="Air temperature in Kelvin"
    )

    process_temperature: float = Field(
        ...,
        description="Process temperature in Kelvin"
    )

    rotational_speed: float = Field(
        ...,
        description="Rotational speed in RPM"
    )

    torque: float = Field(
        ...,
        description="Torque in Newton meters"
    )

    tool_wear: float = Field(
        ...,
        description="Tool wear in minutes"
    )


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def root():
    return {
        "message": "Edge AI Predictive Maintenance API is running",
        "model": "Random Forest",
        "status": "online"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Edge AI Predictive Maintenance",
        "model_loaded": True
    }


# ==========================================
# MODEL INFORMATION
# ==========================================

@app.get("/model-info")
def model_info():
    return {
        "model": "Random Forest Classifier",
        "features": FEATURES,
        "target": "Machine failure",
        "accuracy": 0.9815,
        "roc_auc": 0.9678
    }


# ==========================================
# EXPLAINABILITY ENDPOINT
# ==========================================

@app.get("/explainability")
def explainability():

    importance_data = []

    for feature, importance in zip(
        FEATURES,
        model.feature_importances_
    ):
        importance_data.append({
            "name": feature,
            "importance": round(float(importance), 4),
            "percentage": round(float(importance * 100), 2)
        })

    importance_data.sort(
        key=lambda item: item["importance"],
        reverse=True
    )

    return {
        "model": "Random Forest Classifier",
        "explanation_type": "Global Feature Importance",
        "features": importance_data
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(machine: MachineData):

    input_data = pd.DataFrame([
        {
            "Air temperature [K]": machine.air_temperature,
            "Process temperature [K]": machine.process_temperature,
            "Rotational speed [rpm]": machine.rotational_speed,
            "Torque [Nm]": machine.torque,
            "Tool wear [min]": machine.tool_wear
        }
    ])

    prediction = int(
        model.predict(input_data)[0]
    )

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    failure_percentage = probability * 100

    # ======================================
    # MACHINE STATUS
    # ======================================

    if failure_percentage < 20:

        status = "NORMAL"
        health_score = 100 - failure_percentage

    elif failure_percentage < 50:

        status = "WARNING"
        health_score = 100 - failure_percentage

    else:

        status = "CRITICAL"
        health_score = 100 - failure_percentage

    health_score = round(
        max(0, min(100, health_score)),
        2
    )

    # ======================================
    # RECOMMENDATION
    # ======================================

    if status == "NORMAL":

        recommendation = (
            "Continue operation and monitor machine parameters."
        )

    elif status == "WARNING":

        recommendation = (
            "Schedule maintenance inspection soon."
        )

    else:

        recommendation = (
            "Stop or inspect the machine immediately."
        )

    return {
        "prediction": prediction,
        "status": status,
        "failure_probability": round(
            failure_percentage,
            2
        ),
        "health_score": health_score,
        "recommendation": recommendation,
        "input": {
            "air_temperature": machine.air_temperature,
            "process_temperature": machine.process_temperature,
            "rotational_speed": machine.rotational_speed,
            "torque": machine.torque,
            "tool_wear": machine.tool_wear
        }
    }