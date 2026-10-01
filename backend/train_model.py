import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "data/ai4i2020.csv"

df = pd.read_csv(DATA_PATH)

print("\n========================================")
print("EDGE AI PREDICTIVE MAINTENANCE MODEL")
print("========================================")

print(f"\nDataset loaded successfully.")
print(f"Total records: {len(df)}")


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Machine failure"

X = df[features]
y = df[target]


print("\nFeatures used for training:")

for feature in features:
    print(f" - {feature}")

print(f"\nTarget variable: {target}")


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")


# ==========================================
# 4. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed successfully.")


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print(f"\nROC-AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 8. FEATURE IMPORTANCE
# ==========================================

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

for _, row in importance.iterrows():
    print(
        f"{row['Feature']}: "
        f"{row['Importance']:.4f}"
    )


# ==========================================
# 9. SAVE MODEL
# ==========================================

os.makedirs("model", exist_ok=True)

MODEL_PATH = "model/predictive_maintenance_model.joblib"

joblib.dump(model, MODEL_PATH)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(f"Saved model: {MODEL_PATH}")

print("\nTraining process completed successfully.")