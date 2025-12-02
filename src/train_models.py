"""
Train and evaluate machine learning models for ADHD trial success prediction.

- Uses Leave-One-Out Cross-Validation (LOOCV) for robust metrics.
- Uses Manual Oversampling inside the CV loop to handle class imbalance.
- Generates performance reports and visualizations based on LOOCV results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    balanced_accuracy_score
)
from sklearn.utils import resample


def load_processed_data(file_path: str = "data/processed/adhd_trials_labeled.csv") -> pd.DataFrame:
    """Load processed trial data."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} trials from {file_path}")
    return df


def oversample_minority(X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Manually oversample the minority class to match majority count.
    Essential for datasets with < 5 minority samples.
    """
    # Combine for resampling
    # Ensure X_train is a DataFrame for easier handling
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    train_data = X_train.copy()
    train_data['TARGET'] = y_train

    # Separate classes
    df_class_0 = train_data[train_data.TARGET == 0]
    df_class_1 = train_data[train_data.TARGET == 1]

    # Determine which is minority
    if len(df_class_0) < len(df_class_1):
        df_minority = df_class_0
        df_majority = df_class_1
        n_samples = len(df_class_1)
    else:
        df_minority = df_class_1
        df_majority = df_class_0
        n_samples = len(df_class_0)

    # Upsample minority
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=n_samples,
        random_state=42
    )

    # Combine back
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Split back
    y_upsampled = df_upsampled.TARGET.values
    X_upsampled = df_upsampled.drop('TARGET', axis=1)

    return X_upsampled, y_upsampled


def evaluate_models_loocv(
        X: pd.DataFrame,
        y: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, list]]]:
    """
    Evaluate models using Leave-One-Out Cross-Validation with Oversampling.

    Returns:
    1. Metrics DataFrame (Accuracy, F1, etc.)
    2. Dictionary containing raw LOOCV predictions (for ROC plotting)
    """
    print(f"\n{'=' * 60}")
    print("EVALUATING MODELS (LOOCV + OVERSAMPLING)")
    print(f"{'=' * 60}")

    # Define Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, C=0.5, solver='liblinear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, subsample=0.8,
                                                        random_state=42)
    }

    results_data = []
    model_predictions = {}  # Store y_true, y_prob for ROC curves

    loo = LeaveOneOut()

    for model_name, model in models.items():
        print(f"Testing {model_name}...")

        y_true_all = []
        y_pred_all = []
        y_prob_all = []

        # LOOCV Loop
        # scale inside the loop to prevent data leakage
        for train_index, test_index in loo.split(X):
            X_train_raw, X_test_raw = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns)

            # Oversample (Training data only)
            X_train_res, y_train_res = oversample_minority(X_train_scaled, y_train)

            # Train
            model.fit(X_train_res, y_train_res)

            # Predict
            pred = model.predict(X_test_scaled)[0]
            prob = model.predict_proba(X_test_scaled)[0, 1]

            y_true_all.append(y_test[0])
            y_pred_all.append(pred)
            y_prob_all.append(prob)

        # Calculate Aggregated Metrics
        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_true_all, y_pred_all),
            "Balanced Accuracy": balanced_accuracy_score(y_true_all, y_pred_all),
            "Precision": precision_score(y_true_all, y_pred_all, zero_division=0),
            "Recall": recall_score(y_true_all, y_pred_all, zero_division=0),
            "F1 Score": f1_score(y_true_all, y_pred_all, zero_division=0),
            "AUC": roc_auc_score(y_true_all, y_prob_all)
        }
        results_data.append(metrics)

        # Store for ROC plotting
        model_predictions[model_name] = {
            "y_true": y_true_all,
            "y_prob": y_prob_all
        }

        # Print mini-report
        tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
        print(f"  Balanced Acc: {metrics['Balanced Accuracy']:.1%}")
        print(f"  Failures Caught: {tn}/{tn + fp}")

    results_df = pd.DataFrame(results_data).set_index("Model")

    print(f"\n{'=' * 60}")
    print("LOOCV PERFORMANCE SUMMARY")
    print(f"{'=' * 60}")
    print(results_df.round(3))

    return results_df, model_predictions


def plot_roc_curves_loocv(
        model_predictions: Dict[str, Dict[str, list]],
        save_path: str = "data/processed/roc_curves.png"
):
    """Plot ROC curves using aggregated LOOCV probabilities."""
    plt.figure(figsize=(10, 8))

    for model_name, data in model_predictions.items():
        y_true = data["y_true"]
        y_prob = data["y_prob"]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (LOOCV Aggregated)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nROC curves saved to: {save_path}")
    plt.close()


def generate_feature_importance(
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: list,
        save_path: str = "data/processed/feature_importance.png"
):
    """
    Train a Random Forest on FULL data just to extract feature importance.
    (We can't average feature importance easily across 36 LOOCV models,
    so a full-data fit is the standard proxy for inspection).
    """
    print("\nGenerating Feature Importance Plot (using full dataset fit)...")

    # Scale and Oversample
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_res, y_res = oversample_minority(X_scaled, y)

    # Fit RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_res, y_res)

    # Plot
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 8))
    plt.barh(range(20), importances[indices], align="center", color="steelblue", alpha=0.8)
    plt.yticks(range(20), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()


def save_results(results_df: pd.DataFrame, feature_names: list, save_dir: str = "data/processed"):
    """Save metrics and metadata."""
    os.makedirs(save_dir, exist_ok=True)

    metrics_path = os.path.join(save_dir, "model_performance.csv")
    results_df.to_csv(metrics_path)
    print(f"\nModel performance saved to: {metrics_path}")

    features_path = os.path.join(save_dir, "feature_names.json")
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)


def main():
    print("=" * 60)
    print("ADHD Clinical Trials Model Training & Evaluation")
    print("=" * 60)

    # Load Data
    df = load_processed_data()
    metadata_cols = ["NCTId", "Label", "OverallStatus", "BriefTitle"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X = df[feature_cols]
    y = df["Label"].values

    print(f"\nUsing {len(feature_cols)} features for modeling")

    # Evaluate using LOOCV
    results_df, model_predictions = evaluate_models_loocv(X, y)

    # Generate Visualizations
    plot_roc_curves_loocv(model_predictions)
    generate_feature_importance(X, y, feature_cols)

    # Save Results
    save_results(results_df, feature_cols)

    print("\n" + "=" * 60)
    print("Model training and evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
