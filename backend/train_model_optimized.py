
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_and_prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}\n\n"
            "Make sure ai4i2020.csv is inside backend\\data\\"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {missing}"
        )

    # ------------------------------------------------------------
    # Original machine features
    # ------------------------------------------------------------
    X = df[
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
    ].copy()

    X.columns = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ]

    # ------------------------------------------------------------
    # Product type
    # L = 0, M = 1, H = 2
    # This is an input attribute, not a failure label.
    # ------------------------------------------------------------
    type_values = df["Type"].map(
        {
            "L": 0,
            "M": 1,
            "H": 2,
        }
    )

    if type_values.isna().any():
        raise ValueError(
            "Unexpected values found in the Type column."
        )

    X.insert(
        0,
        "product_type",
        type_values.astype(int),
    )

    # ------------------------------------------------------------
    # Derived features from the sensor inputs.
    # These do not use the target or failure-mode columns.
    # ------------------------------------------------------------
    X["temperature_difference"] = (
        X["process_temperature"]
        - X["air_temperature"]
    )

    X["torque_speed_index"] = (
        X["torque"]
        * X["rotational_speed"]
    )

    X["speed_torque_ratio"] = (
        X["rotational_speed"]
        / (X["torque"] + 1e-6)
    )

    X["wear_torque_index"] = (
        X["tool_wear"]
        * X["torque"]
    )

    y = df["Machine failure"].astype(int)

    return X, y


def create_models():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight=None,
            criterion="gini",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
    }


def ensemble_probability(probabilities, weights):
    return (
        weights["random_forest_weight"]
        * probabilities["random_forest"]
        + weights["hist_gradient_boosting_weight"]
        * probabilities["hist_gradient_boosting"]
        + weights["gradient_boosting_weight"]
        * probabilities["gradient_boosting"]
    )


def find_best_parameters(
    validation_probabilities,
    y_validation,
):
    """
    Select ensemble weights + decision threshold using only
    the validation set.

    Search:
      RF weight : 0.0 to 1.0
      HGB weight: 0.0 to remaining weight
      GB weight : remainder
      Threshold : 0.20 to 0.80

    A fixed grid is used so the result is reproducible.
    """

    best_accuracy = -1.0
    best_parameters = None

    for rf_weight in np.linspace(0.0, 1.0, 11):
        remaining = 1.0 - rf_weight

        for hgb_fraction in np.linspace(0.0, 1.0, 11):
            hgb_weight = remaining * hgb_fraction
            gb_weight = 1.0 - rf_weight - hgb_weight

            probabilities = (
                rf_weight * validation_probabilities["random_forest"]
                + hgb_weight
                * validation_probabilities[
                    "hist_gradient_boosting"
                ]
                + gb_weight
                * validation_probabilities[
                    "gradient_boosting"
                ]
            )

            for threshold in np.linspace(
                0.20,
                0.80,
                121,
            ):
                predictions = (
                    probabilities >= threshold
                ).astype(int)

                current_accuracy = accuracy_score(
                    y_validation,
                    predictions,
                )

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy

                    best_parameters = {
                        "random_forest_weight": float(
                            rf_weight
                        ),
                        "hist_gradient_boosting_weight": float(
                            hgb_weight
                        ),
                        "gradient_boosting_weight": float(
                            gb_weight
                        ),
                        "threshold": float(
                            threshold
                        ),
                    }

    return best_accuracy, best_parameters


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "threshold": float(threshold),
    }


def calculate_permutation_importance(
    models,
    weights,
    threshold,
    X_test,
    y_test,
    random_state=42,
    repeats=5,
):
    """
    Lightweight permutation importance for the final ensemble.

    This is used only after model/threshold selection. It does not
    participate in selecting the final model.
    """

    rng = np.random.default_rng(random_state)

    baseline_probability = ensemble_probability(
        {
            "random_forest": models["random_forest"].predict_proba(
                X_test
            )[:, 1],
            "hist_gradient_boosting": models[
                "hist_gradient_boosting"
            ].predict_proba(X_test)[:, 1],
            "gradient_boosting": models[
                "gradient_boosting"
            ].predict_proba(X_test)[:, 1],
        },
        weights,
    )

    baseline_prediction = (
        baseline_probability >= threshold
    ).astype(int)

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_prediction,
    )

    importances = []

    for feature in X_test.columns:
        drops = []

        for _ in range(repeats):
            X_permuted = X_test.copy()

            values = X_permuted[
                feature
            ].to_numpy(copy=True)

            rng.shuffle(values)

            X_permuted[
                feature
            ] = values

            probability = ensemble_probability(
                {
                    "random_forest": models[
                        "random_forest"
                    ].predict_proba(
                        X_permuted
                    )[:, 1],
                    "hist_gradient_boosting": models[
                        "hist_gradient_boosting"
                    ].predict_proba(
                        X_permuted
                    )[:, 1],
                    "gradient_boosting": models[
                        "gradient_boosting"
                    ].predict_proba(
                        X_permuted
                    )[:, 1],
                },
                weights,
            )

            prediction = (
                probability >= threshold
            ).astype(int)

            permuted_accuracy = accuracy_score(
                y_test,
                prediction,
            )

            drops.append(
                max(
                    baseline_accuracy
                    - permuted_accuracy,
                    0.0,
                )
            )

        importances.append(
            float(np.mean(drops))
        )

    importances = np.asarray(
        importances,
        dtype=float,
    )

    if importances.sum() > 0:
        importances = (
            importances
            / importances.sum()
        )

    return {
        feature: float(value)
        for feature, value in zip(
            X_test.columns,
            importances,
        )
    }


def main():
    print("=" * 72)
    print(
        "EDGE AI PREDICTIVE MAINTENANCE - "
        "OPTIMIZED TRAINING"
    )
    print("=" * 72)

    X, y = load_and_prepare_data()

    print(f"\nDataset records : {len(X)}")
    print(f"Input features  : {len(X.columns)}")
    print(f"Failure records : {int(y.sum())}")
    print(
        f"Normal records  : {int((y == 0).sum())}"
    )

    # ------------------------------------------------------------
    # 60% train / 20% validation / 20% untouched test
    # ------------------------------------------------------------
    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_development,
            y_development,
            test_size=0.25,
            stratify=y_development,
            random_state=43,
        )
    )

    print("\nData split:")
    print(f"Training   : {len(X_train)}")
    print(f"Validation : {len(X_validation)}")
    print(f"Test       : {len(X_test)}")

    models = create_models()

    validation_probabilities = {}
    test_probabilities = {}

    print("\nTraining models...")

    for name, model in models.items():
        print(f"  -> {name}")
        model.fit(
            X_train,
            y_train,
        )

        validation_probabilities[name] = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        test_probabilities[name] = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

    # ------------------------------------------------------------
    # Select ensemble weights + threshold ONLY on validation data
    # ------------------------------------------------------------
    validation_accuracy, parameters = (
        find_best_parameters(
            validation_probabilities,
            y_validation,
        )
    )

    print("\nBest validation configuration:")
    print(
        json.dumps(
            parameters,
            indent=2,
        )
    )

    print(
        f"Validation accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------
    test_probability = ensemble_probability(
        test_probabilities,
        parameters,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probability,
        parameters["threshold"],
    )

    print("\n" + "=" * 72)
    print("FINAL UNTOUCHED TEST RESULTS")
    print("=" * 72)

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )
    print(
        f"ROC-AUC  : "
        f"{test_metrics['roc_auc'] * 100:.2f}%"
    )
    print(
        f"Precision: "
        f"{test_metrics['precision'] * 100:.2f}%"
    )
    print(
        f"Recall   : "
        f"{test_metrics['recall'] * 100:.2f}%"
    )
    print(
        f"F1-score : "
        f"{test_metrics['f1'] * 100:.2f}%"
    )

    print("\nConfusion matrix:")
    print(
        np.asarray(
            test_metrics["confusion_matrix"]
        )
    )

    # ------------------------------------------------------------
    # Refit the selected component models on all 80% development data.
    # The 20% final test set remains untouched.
    # ------------------------------------------------------------
    print(
        "\nRefitting selected components on the "
        "80% development data..."
    )

    production_models = create_models()

    for name, model in production_models.items():
        print(f"  -> refitting {name}")
        model.fit(
            X_development,
            y_development,
        )

    weights = {
        "random_forest_weight": parameters[
            "random_forest_weight"
        ],
        "hist_gradient_boosting_weight": parameters[
            "hist_gradient_boosting_weight"
        ],
        "gradient_boosting_weight": parameters[
            "gradient_boosting_weight"
        ],
    }

    # ------------------------------------------------------------
    # Feature importance for explainability
    # ------------------------------------------------------------
    print(
        "\nCalculating permutation-based "
        "global feature importance..."
    )

    feature_importance = (
        calculate_permutation_importance(
            production_models,
            weights,
            parameters["threshold"],
            X_test,
            y_test,
            random_state=RANDOM_STATE,
            repeats=5,
        )
    )

    print("\nFeature importance:")
    for feature, value in sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"  {feature:30s} "
            f"{value * 100:.2f}%"
        )

    # ------------------------------------------------------------
    # Save model bundle separately from the old website model.
    # ------------------------------------------------------------
    model_bundle = {
        "models": production_models,
        "weights": weights,
        "threshold": parameters["threshold"],
        "feature_names": list(
            X.columns
        ),
        "model_type": (
            "validation-selected weighted ensemble"
        ),
        "dataset": (
            "AI4I 2020 Predictive Maintenance Dataset"
        ),
        "evaluation_protocol": (
            "60/20/20 stratified "
            "train-validation-test split"
        ),
        "target": "Machine failure",
    }

    optimized_model_path = (
        MODEL_DIR
        / "predictive_maintenance_optimized.joblib"
    )

    joblib.dump(
        model_bundle,
        optimized_model_path,
    )

    importance_path = (
        MODEL_DIR
        / "optimized_feature_importance.json"
    )

    with open(
        importance_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_importance,
            file,
            indent=2,
        )

    results = {
        "dataset": (
            "AI4I 2020 Predictive Maintenance Dataset"
        ),
        "features": list(X.columns),
        "model_type": (
            "validation-selected weighted ensemble"
        ),
        "component_models": {
            "random_forest": {
                "n_estimators": 700,
                "max_features": "sqrt",
            },
            "hist_gradient_boosting": {
                "max_iter": 500,
                "learning_rate": 0.04,
                "max_leaf_nodes": 31,
            },
            "gradient_boosting": {
                "n_estimators": 500,
                "learning_rate": 0.03,
                "max_depth": 3,
            },
        },
        "split": {
            "training": int(len(X_train)),
            "validation": int(len(X_validation)),
            "test": int(len(X_test)),
            "stratified": True,
        },
        "validation_accuracy": float(
            validation_accuracy
        ),
        "selected_parameters": parameters,
        "final_test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "base_paper_accuracy": 0.9834,
        "one_percentage_point_target": 0.9934,
    }

    results_path = (
        RESULTS_DIR
        / "optimized_model_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("FILES CREATED")
    print("=" * 72)
    print(optimized_model_path)
    print(importance_path)
    print(results_path)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
