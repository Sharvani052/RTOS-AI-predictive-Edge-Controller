Edge AI Predictive Maintenance Controller for Industry 5.0

An AI-powered predictive maintenance system that analyzes industrial machine operating parameters to predict machine failure, estimate failure risk, explain model behavior, evaluate what-if conditions, and provide maintenance recommendations through a web-based dashboard.

Project Overview

The Edge AI Predictive Maintenance Controller for Industry 5.0 is designed to support proactive machine monitoring using machine learning.

The system uses the AI4I 2020 Predictive Maintenance Dataset and a Random Forest Classifier trained on five machine operating parameters:

Air temperature [K]

Process temperature [K]

Rotational speed [rpm]

Torque [Nm]

Tool wear [min]

The trained model is integrated with a FastAPI backend and a React + Vite frontend. The application provides machine-failure prediction, failure probability, machine health scoring, explainable AI, what-if risk simulation, and maintenance recommendations.

Live Project

🌐 Live Website

https://edge-ai-predictive-maintenance-6zsn.onrender.com

⚙️ Backend API

https://edge-ai-predictive-maintenance-api.onrender.com

📖 FastAPI API Documentation

https://edge-ai-predictive-maintenance-api.onrender.com/docs

💻 GitHub Repository

https://github.com/Sharvani052/RTOS-AI-predictive-Edge-Controller

Main Features

1. Machine Failure Prediction

The system accepts five machine operating parameters and predicts whether the machine is operating normally or has a predicted failure condition.

2. Failure Probability

The Random Forest model produces a failure probability that is displayed as a percentage in the dashboard.

3. Machine Health Score

The application calculates a machine health score using:

Health Score = 100 - Failure Probability

The score is displayed on a scale from 0 to 100.

4. Machine Status

The dashboard converts the predicted failure probability into three application-level operational states:

NORMAL
Failure probability < 20%

WARNING
20% <= Failure probability < 50%

CRITICAL
Failure probability >= 50%

5. Explainable AI

The system uses global feature importance from the trained Random Forest model to show the relative importance of the selected machine parameters.

6. What-If Risk Analysis

The What-If Risk Simulator allows the user to change one or more machine parameters and submit the modified condition to the model.

The system compares the simulated machine condition with the baseline condition and displays the resulting risk information.

7. Maintenance Recommendation

The application provides maintenance-oriented recommendations based on the predicted machine status.

Examples include:

Continue operation and monitor machine parameters.

Schedule maintenance inspection soon.

Stop or inspect the machine immediately.

Machine Learning Model

The project uses a Random Forest Classifier implemented with Scikit-learn.

Model Configuration

Algorithm        : Random Forest Classifier
Number of trees  : 200
Random state     : 42
Class weighting  : Balanced
Train-test split : 80% / 20%
Stratification   : Yes

The class-balanced configuration is used because the machine-failure class is a minority class in the dataset.

Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset.

Dataset Information

Total records    : 10,000
Total columns    : 14
Missing values   : None
Normal records   : 9,661
Failure records  : 339

Selected Features

1. Air temperature [K]
2. Process temperature [K]
3. Rotational speed [rpm]
4. Torque [Nm]
5. Tool wear [min]

Target

Machine failure

where:

0 = Normal operation
1 = Machine failure

Dataset Source

UCI Machine Learning Repository

AI4I 2020 Predictive Maintenance Dataset

DOI:

10.24432/C5HS5C

Model Results

The trained Random Forest classifier was evaluated using a held-out test set containing 2,000 records.

Metric

Result

Accuracy

98.15%

ROC-AUC

96.78%

Failure Precision

73%

Failure Recall

72%

Failure F1-score

73%

Confusion Matrix

                Predicted
              Normal  Failure

Actual Normal    1914      18
Actual Failure     19      49

Therefore:

True Negative  = 1914
False Positive = 18
False Negative = 19
True Positive  = 49

Feature Importance

The trained Random Forest model produced the following global feature-importance values:

Feature

Importance

Torque [Nm]

32.82%

Rotational speed [rpm]

30.62%

Tool wear [min]

22.49%

Air temperature [K]

8.36%

Process temperature [K]

5.71%

System Workflow

The main project workflow is:

User
  ↓
Machine Operating Parameters
  ↓
Data Preprocessing
  ↓
Random Forest Classifier
  ↓
Failure Prediction
  ↓
Failure Probability
  ↓
Machine Health Score
  ↓
Machine Status
  ↓
Maintenance Recommendation

The application also provides explainability and what-if analysis through the dashboard.

What-If Analysis Workflow

Enter Baseline Machine Condition
              ↓
Modify Machine Parameters
              ↓
Run AI Inference
              ↓
Obtain Simulated Result
              ↓
Compare Baseline and Simulated Risk
              ↓
Maintenance Decision Support

Technology Stack

Machine Learning

Python

Pandas

NumPy

Scikit-learn

Joblib

Backend

FastAPI

Uvicorn

Pydantic

Frontend

React

Vite

JavaScript

Recharts

Lucide React

Deployment

GitHub

Render

Project Structure

RTOS-AI-predictive-Edge-Controller/
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
│   ├── inspect_data.py
│   ├── train_model.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
└── .gitignore

Backend API Endpoints

Root Endpoint

GET /

Returns the status of the deployed predictive-maintenance API.

Health Check

GET /health

Returns the backend health and model availability.

Model Information

GET /model-info

Returns model and evaluation information.

Explainability

GET /explainability

Returns global feature-importance information.

Prediction

POST /predict

Accepts the five machine operating parameters.

Example request:

{
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551,
  "torque": 42.8,
  "tool_wear": 0
}

Example response structure:

{
  "prediction": 0,
  "status": "NORMAL",
  "failure_probability": 0.0,
  "health_score": 100,
  "recommendation": "Continue operation and monitor machine parameters."
}

Running the Project Locally

Prerequisites

Install:

Python

Node.js

npm

Git

Clone the Repository

git clone https://github.com/Sharvani052/RTOS-AI-predictive-Edge-Controller.git

Move into the project:

cd RTOS-AI-predictive-Edge-Controller

Start the Backend

cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

Backend:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

Start the Frontend

Open another terminal:

cd frontend
npm install
npm run dev

Frontend:

http://localhost:5173

If port 5173 is already in use, Vite can automatically select another available port.

Train the Model

To retrain the Random Forest model:

cd backend
venv\Scripts\activate
python train_model.py

The trained model is saved as:

backend/model/predictive_maintenance_model.joblib

Deployment

The project is deployed as two services using Render.

Frontend

The React + Vite frontend is deployed as a Render Static Site.

Root Directory : frontend
Build Command  : npm ci && npm run build
Publish Dir.   : dist

Backend

The FastAPI backend is deployed as a Render Web Service.

Root Directory : backend
Build Command  : pip install -r requirements.txt
Start Command  : uvicorn app.main:app --host 0.0.0.0 --port $PORT

Research Context

The project focuses on predictive maintenance using the AI4I 2020 Predictive Maintenance Dataset.

The work combines:

Machine Failure Prediction
        +
Explainable AI
        +
Machine Health Scoring
        +
What-If Risk Analysis
        +
Maintenance Recommendation
        +
Web-Based Deployment

The implemented system uses its own Random Forest configuration and application architecture rather than reproducing the complete experimental setup of the base study.

Limitations

The AI4I 2020 dataset is synthetic.

Five operating parameters are used in the current implementation.

The current evaluation uses one stratified 80--20 train-test split.

Global feature importance provides model-level rather than individual-prediction explanations.

Evaluation with real industrial sensor data is not included in the current implementation.

Future Enhancements

Repeated cross-validation

Additional machine-learning models

Advanced class-imbalance handling

SHAP-based local explanations

Real-time IoT sensor integration

Edge-device deployment

Model compression and optimization

Evaluation on real industrial datasets

Continuous machine monitoring

Team

Arugunta Sharvani
Department of Computer Science and Information Technology
Koneru Lakshmaiah Education Foundation

Muppala Vinusha
Department of Computer Science and Engineering
Koneru Lakshmaiah Education Foundation

Kasarla Anjali
Department of Computer Science and Engineering
Koneru Lakshmaiah Education Foundation

Project Links

Live Website:
https://edge-ai-predictive-maintenance-6zsn.onrender.com

Backend API:
https://edge-ai-predictive-maintenance-api.onrender.com

FastAPI Docs:
https://edge-ai-predictive-maintenance-api.onrender.com/docs

GitHub Repository:
https://github.com/Sharvani052/RTOS-AI-predictive-Edge-Controller

License

This project is developed for academic and research purposes.
