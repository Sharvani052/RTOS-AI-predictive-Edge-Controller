# Edge AI Predictive Maintenance Controller for Industry 5.0

An Edge AI-based predictive maintenance system that uses machine-learning-driven failure prediction, explainable AI, what-if risk analysis, health scoring, and maintenance recommendations to support proactive industrial machine monitoring.

## Project Overview

The **Edge AI Predictive Maintenance Controller for Industry 5.0** is designed to identify the likelihood of machine failure from operational sensor parameters.

The system uses the **AI4I 2020 Predictive Maintenance Dataset** and a **Random Forest Classifier** trained on five machine-related features:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

The trained model predicts the probability of machine failure. The application then converts the prediction into a machine status, health score, and maintenance recommendation.

The project also provides **Explainable AI (XAI)** and a **What-If Risk Simulator**, allowing users to understand important features and test how changing machine parameters affects predicted risk.

---

## Live Project

### Frontend

https://edge-ai-predictive-maintenance-6zsn.onrender.com

### Backend API

https://edge-ai-predictive-maintenance-api.onrender.com

### API Documentation

https://edge-ai-predictive-maintenance-api.onrender.com/docs

### GitHub Repository

https://github.com/Sharvani052/RTOS-AI-predictive-Edge-Controller

---

## Key Features

### 1. Machine Failure Prediction

The system accepts five machine operating parameters and predicts the probability of machine failure using a trained Random Forest model.

### 2. Machine Health Score

A health score is calculated from the predicted failure probability.

The application uses:

- **NORMAL**: failure probability < 20%
- **WARNING**: failure probability from 20% to below 50%
- **CRITICAL**: failure probability >= 50%

The health score is calculated as:

```text
Health Score = 100 - Failure Probability (%)
```

The resulting score is bounded between 0 and 100.

### 3. Maintenance Recommendation

The application generates a recommendation according to the predicted status:

| Status | Recommendation |
|---|---|
| NORMAL | Continue operation and monitor |
| WARNING | Schedule maintenance inspection soon |
| CRITICAL | Stop or inspect immediately |

### 4. Explainable AI

The system provides feature-importance information to help users understand which input parameters contribute most to the trained Random Forest model.

### 5. What-If Risk Simulator

Users can modify machine parameters and analyze how changes in temperature, rotational speed, torque, or tool wear affect the predicted failure risk.

### 6. Interactive Web Dashboard

The React-based dashboard provides:

- Machine parameter input
- Failure probability
- Health score
- Machine status
- Maintenance recommendation
- Explainability information
- What-if analysis

---

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

Dataset file:

```text
ai4i2020.csv
```

Dataset location in the project:

```text
backend/data/ai4i2020.csv
```

### Dataset Summary

- Total records: **10,000**
- Total columns: **14**
- Missing values: **None**
- Target column: **Machine failure**

### Class Distribution

| Class | Records |
|---|---:|
| Normal / No Failure | 9,661 |
| Machine Failure | 339 |
| Total | 10,000 |

Because machine failures represent a much smaller portion of the dataset, the Random Forest model uses class balancing during training.

---

## Machine Learning Model

### Algorithm

**Random Forest Classifier**

### Configuration

```text
Number of Trees: 200
Random State: 42
Class Weight: balanced
Training/Testing Split: 80/20
Split Type: Stratified
```

The model is saved as:

```text
backend/model/predictive_maintenance_model.joblib
```

### Input Features

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

### Target

```text
Machine failure
```

---

## Model Performance

The trained model produced the following evaluation results on the test set:

| Metric | Result |
|---|---:|
| Accuracy | **98.15%** |
| ROC-AUC | **96.78%** |
| Failure Precision | **73%** |
| Failure Recall | **72%** |
| Failure F1-score | **73%** |

### Confusion Matrix

```text
                     Predicted
                  Normal   Failure
Actual Normal      1914       18
Actual Failure       19       49
```

This evaluation is based on the 20% stratified test split.

---

## Feature Importance

The Random Forest model provides the following global feature-importance values:

| Feature | Importance |
|---|---:|
| Torque [Nm] | 32.82% |
| Rotational speed [rpm] | 30.62% |
| Tool wear [min] | 22.49% |
| Air temperature [K] | 8.36% |
| Process temperature [K] | 5.71% |

These values represent the contribution of each feature to the trained Random Forest model's decision process.

---

## System Architecture

```text
                ┌─────────────────────────┐
                │      React + Vite       │
                │     Web Dashboard       │
                └────────────┬────────────┘
                             │
                             │ HTTP / REST API
                             ▼
                ┌─────────────────────────┐
                │      FastAPI Backend    │
                │       /predict          │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │   Random Forest Model   │
                │  predictive_maintenance │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Prediction + Risk +     │
                │ Health + Recommendation │
                └─────────────────────────┘
```

---

## Prediction Workflow

```text
Machine Parameters
        │
        ▼
Feature Validation
        │
        ▼
Random Forest Prediction
        │
        ▼
Failure Probability
        │
        ├───────────────┐
        │               │
        ▼               ▼
  Health Score      Status
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
           NORMAL     WARNING    CRITICAL
                        │
                        ▼
             Maintenance Recommendation
```

---

## What-If Analysis Workflow

```text
Current Machine Parameters
             │
             ▼
       Change Parameters
             │
             ▼
      Send New Input Data
             │
             ▼
      Random Forest Model
             │
             ▼
   Compare Predicted Risk
             │
             ▼
   Display Updated Machine Status
```

---

## Project Structure

```text
Edge-AI-Predictive-Maintenance/
│
├── backend/
│   ├── app/
│   │   └── main.py
│   │
│   ├── data/
│   │   └── ai4i2020.csv
│   │
│   ├── model/
│   │   └── predictive_maintenance_model.joblib
│   │
│   ├── requirements.txt
│   └── venv/
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── ...
│   ├── package.json
│   └── ...
│
└── README.md
```

---

## Technology Stack

### Frontend

- React
- Vite
- JavaScript
- React Router
- Recharts
- Lucide React

### Backend

- Python
- FastAPI
- Uvicorn

### Machine Learning

- Pandas
- NumPy
- Scikit-learn
- Joblib

### Model

- Random Forest Classifier

### Deployment

- Render
- GitHub

---

## Backend API

### `GET /`

Checks that the API is running.

### `GET /health`

Returns the health status of the backend service.

### `GET /model-info`

Returns information about the loaded machine-learning model.

### `GET /explainability`

Returns global feature-importance information used by the explainability section.

### `POST /predict`

Accepts machine parameters and returns a prediction.

Example request:

```json
{
  "air_temperature": 300,
  "process_temperature": 310,
  "rotational_speed": 1200,
  "torque": 65,
  "tool_wear": 200
}
```

The response includes prediction-related information such as:

```text
prediction
status
failure_probability
health_score
recommendation
```

---

## Local Setup

### Prerequisites

Install:

- Python
- Node.js
- npm
- Git

---

## Backend Setup

Open PowerShell and move to the backend directory:

```powershell
cd D:\Edge-AI-Predictive-Maintenance\backend
```

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the FastAPI server:

```powershell
uvicorn app.main:app --reload
```

The backend will normally be available at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Frontend Setup

Open another terminal and move to:

```powershell
cd D:\Edge-AI-Predictive-Maintenance\frontend
```

Install the required packages:

```powershell
npm install
```

Start the Vite development server:

```powershell
npm run dev
```

The frontend will normally open at a local Vite address such as:

```text
http://localhost:5173
```

---

## Training the Model

The training process uses the AI4I 2020 dataset and selected machine features.

The general training workflow is:

```text
Load Dataset
    │
    ▼
Select Five Features
    │
    ▼
Select Machine Failure Target
    │
    ▼
Stratified 80/20 Train-Test Split
    │
    ▼
Train Random Forest
    │
    ▼
Evaluate Model
    │
    ▼
Save Model as .joblib
```

The trained model is then loaded by the FastAPI backend for prediction.

---

## Deployment

### Backend Deployment

The backend is deployed using Render.

Backend root directory:

```text
backend
```

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Frontend Deployment

The frontend is deployed as a Render static site.

Frontend root directory:

```text
frontend
```

Build command:

```text
npm ci && npm run build
```

Publish directory:

```text
dist
```

The frontend communicates with the deployed FastAPI backend through the configured API URL.

---

## Research Relevance

The project combines predictive maintenance with an application-oriented edge/cloud deployment workflow.

The implementation focuses on:

- Machine failure prediction
- Risk interpretation
- Explainable AI
- What-if analysis
- Health scoring
- Maintenance recommendations
- Interactive monitoring

The system is intended as a practical prototype for proactive machine monitoring in an Industry 5.0-oriented environment.

---

## Limitations

- The model is trained using the AI4I 2020 dataset rather than live industrial sensor streams.
- The five selected input features represent only a subset of possible machine condition variables.
- Predictions depend on the distribution and characteristics of the training dataset.
- The application-level health thresholds are rule-based.
- The current implementation does not directly control physical industrial equipment.

---

## Future Enhancements

Possible future extensions include:

- Real-time IoT sensor integration
- Edge-device deployment
- Continuous model retraining
- Additional industrial sensor features
- Time-series-based predictive maintenance
- SHAP-based local explanations
- Multi-machine monitoring
- Alert and notification systems
- Maintenance history integration
- Digital twin integration

---

## Team

### 1. Arugunta Sharvani
2420090052

Computer Science and Information Technology

### 2. Muppala Vinusha
2420030212

Computer Science and Engineering

### 3. Kasarla Anjali
2420030170

Computer Science and Engineering

---

## License

This project is developed for academic and research purposes.

---

## Acknowledgement

The project uses the AI4I 2020 Predictive Maintenance Dataset for machine-failure prediction experiments.
