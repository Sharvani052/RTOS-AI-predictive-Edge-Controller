from pathlib import Path
import json
from typing import Union

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"

OPTIMIZED_MODEL_PATH = (
    MODEL_DIR / "predictive_maintenance_optimized.joblib"
)

IMPORTANCE_PATH = (
    MODEL_DIR / "optimized_feature_importance.json"
)

RESULTS_PATH = (
    BASE_DIR / "results" / "optimized_model_results.json"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Edge AI Predictive Maintenance Controller",
    description=(
        "Optimized AI inference API for the "
        "Edge AI Predictive Maintenance Controller for Industry 5.0."
    ),
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://edge-ai-predictive-maintenance-6zsn.onrender.com",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INPUT MODEL
# ============================================================

class MachineInput(BaseModel):
    air_temperature: float = Field(
        ...,
        description="Air temperature in Kelvin",
    )

    process_temperature: float = Field(
        ...,
        description="Process temperature in Kelvin",
    )

    rotational_speed: float = Field(
        ...,
        description="Rotational speed in RPM",
    )

    torque: float = Field(
        ...,
        description="Torque in Newton meters",
    )

    tool_wear: float = Field(
        ...,
        description="Tool wear in minutes",
    )

    product_type: Union[str, int] = Field(
        default="M",
        description=(
            "Product type: L, M, H or 0, 1, 2. "
            "Defaults to M for compatibility with the existing frontend."
        ),
    )


# ============================================================
# GLOBAL MODEL OBJECTS
# ============================================================

model_bundle = None
feature_importance = {}
model_results = {}


# ============================================================
# LOAD OPTIMIZED MODEL
# ============================================================

def load_model_bundle():
    global model_bundle

    if not OPTIMIZED_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Optimized model was not found:\n"
            f"{OPTIMIZED_MODEL_PATH}\n\n"
            "Run train_model_optimized.py first."
        )

    model_bundle = joblib.load(
        OPTIMIZED_MODEL_PATH
    )


# ============================================================
# LOAD EXPLAINABILITY DATA
# ============================================================

def load_explainability_data():
    global feature_importance

    if not IMPORTANCE_PATH.exists():
        feature_importance = {}
        return

    with open(
        IMPORTANCE_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        feature_importance = json.load(file)


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

def load_model_results():
    global model_results

    if not RESULTS_PATH.exists():
        model_results = {}
        return

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        model_results = json.load(file)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    load_model_bundle()
    load_explainability_data()
    load_model_results()


# ============================================================
# PRODUCT TYPE ENCODING
# ============================================================

def encode_product_type(
    product_type: Union[str, int]
) -> int:

    if isinstance(product_type, str):
        value = product_type.strip().upper()

        mapping = {
            "L": 0,
            "M": 1,
            "H": 2,
            "0": 0,
            "1": 1,
            "2": 2,
        }

        if value in mapping:
            return mapping[value]

    elif isinstance(product_type, int):
        if product_type in [0, 1, 2]:
            return product_type

    raise ValueError(
        "Invalid product_type. Use L, M, H or 0, 1, 2."
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    machine: MachineInput
) -> pd.DataFrame:

    product_type = encode_product_type(
        machine.product_type
    )

    air_temperature = float(
        machine.air_temperature
    )

    process_temperature = float(
        machine.process_temperature
    )

    rotational_speed = float(
        machine.rotational_speed
    )

    torque = float(
        machine.torque
    )

    tool_wear = float(
        machine.tool_wear
    )

    temperature_difference = (
        process_temperature
        - air_temperature
    )

    torque_speed_index = (
        torque
        * rotational_speed
    )

    speed_torque_ratio = (
        rotational_speed
        / (torque + 1e-6)
    )

    wear_torque_index = (
        tool_wear
        * torque
    )

    data = {
        "product_type": product_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "temperature_difference": temperature_difference,
        "torque_speed_index": torque_speed_index,
        "speed_torque_ratio": speed_torque_ratio,
        "wear_torque_index": wear_torque_index,
    }

    X = pd.DataFrame(
        [data]
    )

    # Keep exactly the feature order used during training.
    if model_bundle is not None:
        trained_features = model_bundle.get(
            "feature_names",
            [],
        )

        if trained_features:
            missing_features = [
                feature
                for feature in trained_features
                if feature not in X.columns
            ]

            if missing_features:
                raise ValueError(
                    "Missing features required by model: "
                    f"{missing_features}"
                )

            X = X[
                trained_features
            ]

    return X


# ============================================================
# ENSEMBLE PROBABILITY
# ============================================================

def calculate_ensemble_probability(
    X: pd.DataFrame
) -> float:

    if model_bundle is None:
        raise RuntimeError(
            "Optimized model is not loaded."
        )

    models = model_bundle["models"]
    weights = model_bundle["weights"]

    probabilities = {}

    for name, model in models.items():

        probability = model.predict_proba(
            X
        )[0][1]

        probabilities[name] = float(
            probability
        )

    ensemble_probability = (
        weights["random_forest_weight"]
        * probabilities["random_forest"]
        +
        weights["hist_gradient_boosting_weight"]
        * probabilities["hist_gradient_boosting"]
        +
        weights["gradient_boosting_weight"]
        * probabilities["gradient_boosting"]
    )

    return float(
        np.clip(
            ensemble_probability,
            0.0,
            1.0,
        )
    )


# ============================================================
# STATUS CALCULATION
# ============================================================

def calculate_status(
    failure_percentage: float
) -> str:

    if failure_percentage < 20:
        return "NORMAL"

    if failure_percentage < 50:
        return "WARNING"

    return "CRITICAL"


# ============================================================
# RECOMMENDATION
# ============================================================

def get_recommendation(
    status: str
) -> str:

    if status == "NORMAL":
        return (
            "Continue operation and monitor "
            "machine parameters."
        )

    if status == "WARNING":
        return (
            "Schedule maintenance inspection soon."
        )

    return (
        "Stop or inspect the machine immediately."
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": (
            "Edge AI Predictive Maintenance "
            "Controller API is running."
        ),
        "model": (
            "Optimized validation-selected "
            "weighted ensemble"
        ),
        "accuracy": "99.45%",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    model_available = (
        model_bundle is not None
    )

    return {
        "status": (
            "healthy"
            if model_available
            else "unhealthy"
        ),
        "model_loaded": model_available,
        "model_file": (
            OPTIMIZED_MODEL_PATH.name
        ),
    }


# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

@app.get("/model-info")
def model_info():

    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Optimized model is not loaded.",
        )

    return {
        "model_type": model_bundle.get(
            "model_type",
            "validation-selected weighted ensemble",
        ),
        "dataset": model_bundle.get(
            "dataset",
            "AI4I 2020 Predictive Maintenance Dataset",
        ),
        "target": model_bundle.get(
            "target",
            "Machine failure",
        ),
        "features": model_bundle.get(
            "feature_names",
            [],
        ),
        "evaluation_protocol": model_bundle.get(
            "evaluation_protocol",
            "60/20/20 stratified train-validation-test split",
        ),
        "accuracy": model_results.get(
            "final_test_metrics",
            {}
        ).get(
            "accuracy",
            0.9945,
        ),
        "accuracy_percent": (
            model_results.get(
                "final_test_metrics",
                {}
            ).get(
                "accuracy",
                0.9945,
            )
            * 100
        ),
        "roc_auc": model_results.get(
            "final_test_metrics",
            {}
        ).get(
            "roc_auc",
            0.9571,
        ),
        "roc_auc_percent": (
            model_results.get(
                "final_test_metrics",
                {}
            ).get(
                "roc_auc",
                0.9571,
            )
            * 100
        ),
    }


# ============================================================
# EXPLAINABILITY ENDPOINT
# ============================================================

@app.get("/explainability")
def explainability():

    features = []

    for name, value in sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        percentage = float(value) * 100

        features.append(
            {
                "name": name,
                "value": float(value),
                "percentage": percentage,
            }
        )

    return {
        "model": (
            "Validation-Selected "
            "Weighted Ensemble"
        ),
        "explanation_type": (
            "Permutation-Based Global "
            "Feature Importance"
        ),
        "features": features,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    machine: MachineInput
):

    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Optimized model is not loaded. "
                "Check the backend startup logs."
            ),
        )

    try:
        X = build_features(
            machine
        )

        failure_probability = (
            calculate_ensemble_probability(
                X
            )
        )

        threshold = float(
            model_bundle.get(
                "threshold",
                0.47,
            )
        )

        prediction = int(
            failure_probability >= threshold
        )

        failure_percentage = (
            failure_probability
            * 100
        )

        health_score = (
            100
            - failure_percentage
        )

        health_score = float(
            np.clip(
                health_score,
                0.0,
                100.0,
            )
        )

        status = calculate_status(
            failure_percentage
        )

        recommendation = (
            get_recommendation(
                status
            )
        )

        return {
            "prediction": prediction,
            "status": status,
            "failure_probability": round(
                failure_percentage,
                2,
            ),
            "health_score": round(
                health_score,
                2,
            ),
            "recommendation": recommendation,
            "threshold": threshold,
            "model": (
                "Optimized Weighted Ensemble"
            ),
            "product_type": str(
                machine.product_type
            ),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(error)}"
            ),
        )


# ============================================================
# LOCAL ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )