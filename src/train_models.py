"""
Train and evaluate machine learning models for ADHD trial success prediction.

This module:
1. Loads processed trial data
2. Splits into train/test sets
3. Trains multiple ML models
4. Evaluates and compares model performance
5. Generates visualizations and saves results
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
    classification_report,
)


def load_processed_data(file_path: str = "data/processed/adhd_trials_labeled.csv") -> pd.DataFrame:
    """
    Load processed trial data.

    Parameters
    ----------
    file_path : str
        Path to processed data CSV

    Returns
    -------
    pd.DataFrame
        Processed trial data
    """
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} trials from {file_path}")
    return df


def prepare_train_test_split(
    df: pd.DataFrame,
    feature_cols: list,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index, pd.Index]:
    """
    Prepare train/test split with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Processed trial data
    feature_cols : list
        List of feature column names
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    Tuple containing X_train, X_test, y_train, y_test, train_idx, test_idx
    """
    X = df[feature_cols].values
    y = df["Label"].values

    # Split with stratification
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n{'='*60}")
    print("TRAIN/TEST SPLIT")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Success: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    print(f"  Failure: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"\nTest samples: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
    print(f"  Success: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
    print(f"  Failure: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")

    return X_train, X_test, y_train, y_test, train_idx, test_idx


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features using training set statistics.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Test features

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, StandardScaler]
        Scaled training features, scaled test features, fitted scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nFeatures standardized (mean=0, std=1)")

    return X_train_scaled, X_test_scaled, scaler


def train_models(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> Dict[str, Any]:
    """
    Train multiple classification models.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    random_state : int
        Random seed

    Returns
    -------
    Dict[str, Any]
        Dictionary of trained models
    """
    print(f"\n{'='*60}")
    print("TRAINING MODELS")
    print(f"{'='*60}")

    models = {}

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(random_state=random_state, max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=random_state,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state
    )
    gb.fit(X_train, y_train)
    models["Gradient Boosting"] = gb

    print(f"\nTrained {len(models)} models")

    return models


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """
    Evaluate a single model on train and test sets.

    Parameters
    ----------
    model : Any
        Trained model
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model

    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
        "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
        "train_auc": roc_auc_score(y_train, y_train_proba),
        "test_auc": roc_auc_score(y_test, y_test_proba),
    }

    return metrics


def evaluate_all_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Evaluate all models and create comparison table.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of trained models
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels

    Returns
    -------
    pd.DataFrame
        Comparison table of model performance
    """
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")

    results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
        results[model_name] = metrics

    # Create comparison DataFrame
    results_df = pd.DataFrame(results).T

    # Reorder columns
    col_order = [
        "test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc",
        "train_accuracy", "train_precision", "train_recall", "train_f1", "train_auc"
    ]
    results_df = results_df[col_order]

    print(f"\n{'='*60}")
    print("MODEL COMPARISON (Test Set)")
    print(f"{'='*60}")
    print(results_df[[c for c in col_order if c.startswith("test")]].round(3))

    return results_df


def plot_roc_curves(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "data/processed/roc_curves.png"
):
    """
    Plot ROC curves for all models.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of trained models
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    save_path : str
        Path to save plot
    """
    plt.figure(figsize=(10, 8))

    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random Classifier")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - ADHD Trial Success Prediction", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nROC curves saved to: {save_path}")
    plt.close()


def plot_feature_importance(
    model: Any,
    feature_names: list,
    model_name: str,
    top_n: int = 20,
    save_path: str = "data/processed/feature_importance.png"
):
    """
    Plot feature importance for tree-based models.

    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
    top_n : int
        Number of top features to display
    save_path : str
        Path to save plot
    """
    if not hasattr(model, "feature_importances_"):
        print(f"\n{model_name} does not have feature importances")
        return

    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(
        range(top_n),
        importances[indices],
        align="center",
        alpha=0.8,
        color="steelblue"
    )
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Top {top_n} Feature Importances - {model_name}", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    save_path = save_path.replace(".png", f"_{model_name.replace(' ', '_').lower()}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    save_path: str = "data/processed/confusion_matrix.png"
):
    """
    Plot confusion matrix for a model.

    Parameters
    ----------
    model : Any
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model
    save_path : str
        Path to save plot
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.colorbar()

    classes = ["Failure (0)", "Success (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16, fontweight="bold")

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    save_path = save_path.replace(".png", f"_{model_name.replace(' ', '_').lower()}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def save_results(
    results_df: pd.DataFrame,
    models: Dict[str, Any],
    feature_names: list,
    save_dir: str = "data/processed"
):
    """
    Save model results and metadata.

    Parameters
    ----------
    results_df : pd.DataFrame
        Model performance comparison
    models : Dict[str, Any]
        Trained models
    feature_names : list
        List of feature names
    save_dir : str
        Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save performance metrics
    metrics_path = os.path.join(save_dir, "model_performance.csv")
    results_df.to_csv(metrics_path)
    print(f"\nModel performance saved to: {metrics_path}")

    # Save feature names
    features_path = os.path.join(save_dir, "feature_names.json")
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved to: {features_path}")


def main():
    """Main execution function."""
    print("="*60)
    print("ADHD Clinical Trials Model Training")
    print("="*60)

    # Load processed data
    df = load_processed_data()

    # Get feature columns (exclude metadata)
    metadata_cols = ["NCTId", "Label", "OverallStatus", "BriefTitle"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    print(f"\nUsing {len(feature_cols)} features for modeling")

    # Prepare train/test split
    X_train, X_test, y_train, y_test, train_idx, test_idx = prepare_train_test_split(
        df, feature_cols, test_size=0.2, random_state=42
    )

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train models
    models = train_models(X_train_scaled, y_train, random_state=42)

    # Evaluate models
    results_df = evaluate_all_models(models, X_train_scaled, y_train, X_test_scaled, y_test)

    # Generate visualizations
    plot_roc_curves(models, X_test_scaled, y_test)

    # Plot feature importance for tree-based models
    plot_feature_importance(models["Random Forest"], feature_cols, "Random Forest")
    plot_feature_importance(models["Gradient Boosting"], feature_cols, "Gradient Boosting")

    # Plot confusion matrices for best model
    best_model_name = results_df["test_auc"].idxmax()
    best_model = models[best_model_name]
    print(f"\nBest model by AUC: {best_model_name}")

    plot_confusion_matrix(best_model, X_test_scaled, y_test, best_model_name)

    # Save results
    save_results(results_df, models, feature_cols)

    print("\n" + "="*60)
    print("Model training complete!")
    print("="*60)

    return models, results_df


if __name__ == "__main__":
    main()
