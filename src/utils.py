"""
Utility functions for ADHD clinical trials prediction pipeline.

This module contains helper functions for data analysis, visualization,
and model interpretation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    summary = pd.DataFrame({
        "Type": df.dtypes,
        "Missing": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2),
        "Unique": df.nunique(),
        "Sample Value": df.iloc[0]
    })

    return summary


def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str = "Label",
    save_path: str = None
) -> None:
    """
    Plot the distribution of class labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    label_col : str
        Name of label column
    save_path : str, optional
        Path to save plot
    """
    plt.figure(figsize=(8, 6))

    counts = df[label_col].value_counts()
    labels = ["Failure (0)", "Success (1)"]
    colors = ["#e74c3c", "#2ecc71"]

    plt.bar(range(len(counts)), counts.values, color=colors, alpha=0.8, edgecolor="black")
    plt.xticks(range(len(counts)), labels, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Trial Outcomes", fontsize=14, fontweight="bold")

    # Add count labels on bars
    for i, v in enumerate(counts.values):
        plt.text(i, v + 5, str(v), ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Add percentage labels
    total = counts.sum()
    for i, v in enumerate(counts.values):
        pct = v / total * 100
        plt.text(i, v / 2, f"{pct:.1f}%", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Class distribution plot saved to: {save_path}")

    plt.close()


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "Label",
    save_path: str = None,
    n_cols: int = 4
) -> None:
    """
    Plot distributions of features by class.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : List[str]
        List of feature columns to plot
    label_col : str
        Name of label column
    save_path : str, optional
        Path to save plot
    n_cols : int
        Number of columns in subplot grid
    """
    # Select only numeric features for distribution plots
    numeric_features = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64]]

    n_features = min(len(numeric_features), 16)  # Limit to 16 features
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, col in enumerate(numeric_features[:n_features]):
        ax = axes[idx]

        # Plot distributions for each class
        for label in sorted(df[label_col].unique()):
            data = df[df[label_col] == label][col].dropna()
            label_name = "Success" if label == 1 else "Failure"
            color = "#2ecc71" if label == 1 else "#e74c3c"

            ax.hist(data, alpha=0.6, label=label_name, bins=20, color=color, edgecolor="black")

        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Feature Distributions by Outcome", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Feature distributions plot saved to: {save_path}")

    plt.close()


def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot correlation matrix of features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : List[str]
        List of feature columns
    save_path : str, optional
        Path to save plot
    figsize : Tuple[int, int]
        Figure size
    """
    # Select only numeric features
    numeric_features = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64]]

    corr_matrix = df[numeric_features].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Correlation matrix plot saved to: {save_path}")

    plt.close()


def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing data patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Missing data analysis
    """
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": df.isnull().sum().values,
        "Missing_Percentage": (df.isnull().sum() / len(df) * 100).values
    })

    missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values(
        "Missing_Percentage", ascending=False
    )

    return missing_df


def get_top_features_by_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Get top N most important features from a model.

    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ attribute
    feature_names : List[str]
        List of feature names
    top_n : int
        Number of top features to return

    Returns
    -------
    pd.DataFrame
        Top features with importance scores
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature importances")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = pd.DataFrame({
        "Feature": [feature_names[i] for i in indices],
        "Importance": importances[indices]
    })

    return top_features


def print_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> None:
    """
    Print a formatted classification summary.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    model_name : str
        Name of the model
    """
    from sklearn.metrics import classification_report

    print(f"\n{'='*60}")
    print(f"{model_name} - Classification Report")
    print(f"{'='*60}")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Failure (0)", "Success (1)"],
        digits=3
    ))


def calculate_baseline_metrics(y_true: np.ndarray) -> Dict[str, float]:
    """
    Calculate baseline metrics for comparison.

    Parameters
    ----------
    y_true : np.ndarray
        True labels

    Returns
    -------
    Dict[str, float]
        Baseline metrics
    """
    # Majority class baseline
    majority_class = 1 if (y_true == 1).sum() > (y_true == 0).sum() else 0
    majority_pred = np.full_like(y_true, majority_class)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    baseline = {
        "accuracy": accuracy_score(y_true, majority_pred),
        "precision": precision_score(y_true, majority_pred, zero_division=0),
        "recall": recall_score(y_true, majority_pred, zero_division=0),
        "f1": f1_score(y_true, majority_pred, zero_division=0),
    }

    return baseline


def create_summary_table(
    df: pd.DataFrame,
    group_by: str,
    label_col: str = "Label"
) -> pd.DataFrame:
    """
    Create a summary table grouped by a specific column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    group_by : str
        Column to group by
    label_col : str
        Label column name

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    summary = df.groupby(group_by).agg({
        label_col: ["count", "sum", "mean"]
    }).round(3)

    summary.columns = ["Total_Trials", "Successes", "Success_Rate"]
    summary = summary.sort_values("Total_Trials", ascending=False)

    return summary


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds

    Returns
    -------
    str
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


def save_dataframe_summary(df: pd.DataFrame, save_path: str) -> None:
    """
    Save a comprehensive summary of a dataframe to a text file.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    save_path : str
        Path to save summary
    """
    with open(save_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("DATASET SUMMARY\n")
        f.write("="*60 + "\n\n")

        f.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")

        f.write("Column Information:\n")
        f.write("-"*60 + "\n")
        summary = get_data_summary(df)
        f.write(summary.to_string())
        f.write("\n\n")

        f.write("Descriptive Statistics:\n")
        f.write("-"*60 + "\n")
        f.write(df.describe().to_string())
        f.write("\n\n")

    print(f"Dataset summary saved to: {save_path}")
