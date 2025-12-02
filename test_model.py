"""
Test and validate the ADHD clinical trials prediction model.

Strategy since the dataset is small:
1. Uses Leave-One-Out Cross-Validation (LOOCV) instead of K-Fold.
   - Essential for N=36. We train on 35, test on 1, repeat 36 times.
2. Implements Manual Random Oversampling.
   - Inside every training fold, we duplicate the minority class (failures)
     until they equal the majority class. This forces the model to learn them.
3. Uses Balanced Accuracy.
   - Standard accuracy is misleading. Balanced Accuracy is the average of
     Sensitivity (Success accuracy) and Specificity (Failure accuracy).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.utils import resample
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def load_data():
    """Load processed data."""
    path = 'data/processed/adhd_trials_labeled.csv'
    if not pd.io.common.file_exists(path):
        print(f"Error: {path} not found. Please run prepare_data.py first.")
        return None, None, None, None

    df = pd.read_csv(path)

    metadata_cols = ['NCTId', 'Label', 'OverallStatus', 'BriefTitle']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    X = df[feature_cols]
    y = df['Label'].values

    return df, X, y, feature_cols


def oversample_minority(X_train, y_train):
    """
    Manually oversample the minority class to match majority count.
    This is critical for datasets where minority N < 5.
    """
    # Combine for resampling
    train_data = X_train.copy()
    train_data['TARGET'] = y_train

    # Separate classes
    count_class_0, count_class_1 = train_data.TARGET.value_counts().sort_index()

    df_class_0 = train_data[train_data.TARGET == 0]
    df_class_1 = train_data[train_data.TARGET == 1]

    # Determine which is minority
    if len(df_class_0) < len(df_class_1):
        # Class 0 (Failure) is minority
        df_minority = df_class_0
        df_majority = df_class_1
        n_samples = len(df_class_1)
    else:
        # Class 1 (Success) is minority (unlikely here)
        df_minority = df_class_1
        df_majority = df_class_0
        n_samples = len(df_class_0)

    # Upsample minority
    df_minority_upsampled = resample(
        df_minority,
        replace=True,  # Sample with replacement
        n_samples=n_samples,  # Match majority class
        random_state=42
    )

    # Combine back
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Split back into X and y
    y_upsampled = df_upsampled.TARGET.values
    X_upsampled = df_upsampled.drop('TARGET', axis=1)

    return X_upsampled, y_upsampled


def test_loocv_with_oversampling():
    """
    Perform Leave-One-Out Cross-Validation with Oversampling.
    """
    print("=" * 70)
    print("1. LEAVE-ONE-OUT CROSS-VALIDATION (With Oversampling)")
    print("=" * 70)
    print("Strategy: Train on 35, Test on 1. Repeat 36 times.")
    print("Correction: Inside every training loop, we duplicate the Failures")
    print("            so the model sees 50/50 Success/Failure.\n")

    df, X, y, feature_cols = load_data()
    if df is None: return

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Models
    models = {
        'Baseline (Dummy)': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(random_state=42, C=0.5, solver='liblinear'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, subsample=0.8,
                                                        random_state=42)
    }

    loo = LeaveOneOut()

    best_model_name = ""
    best_balanced_acc = -1

    for name, model in models.items():
        print(f"Testing {name}...")

        y_true_all = []
        y_pred_all = []

        # Manual LOOCV Loop
        for train_index, test_index in loo.split(X_scaled):
            X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # --- CRITICAL STEP: OVERSAMPLE TRAINING DATA ---
            # We only touch training data. Test data remains pure.
            X_train_res, y_train_res = oversample_minority(X_train, y_train)

            # Train
            model.fit(X_train_res, y_train_res)

            # Predict
            pred = model.predict(X_test)[0]

            y_true_all.append(y_test[0])
            y_pred_all.append(pred)

        # Calculate Metrics on the aggregated predictions
        acc = accuracy_score(y_true_all, y_pred_all)
        bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
        cm = confusion_matrix(y_true_all, y_pred_all)

        print(f"  Raw Accuracy:      {acc:.1%}")
        print(f"  Balanced Accuracy: {bal_acc:.1%}")
        try:
            tn, fp, fn, tp = cm.ravel()
            print(f"  Confusion Matrix:  [TN={tn}, FP={fp}] (Failures)")
            print(f"                     [FN={fn}, TP={tp}] (Successes)")

            # Check if we caught any failures
            if tn > 0:
                print(f"  >> SUCCESS: Model identified {tn} failures correctly!")
            else:
                if name != 'Baseline (Dummy)':
                    print(f"  >> WARNING: Model still missed all failures.")
                else:
                    print(f"  >> INFO: Baseline naturally misses all failures.")

        except:
            print(f"  Confusion Matrix: \n{cm}")

        # Track best model (excluding dummy)
        if name != 'Baseline (Dummy)' and bal_acc > best_balanced_acc:
            best_balanced_acc = bal_acc
            best_model_name = name
        print("-" * 30)

    print(f"\nBEST MODEL: {best_model_name} (Balanced Acc: {best_balanced_acc:.1%})")
    return best_model_name


def analyze_feature_importance_on_failures():
    """
    Since we only have 3 failures, let's look at them directly
    compared to the average success.
    """
    print("\n" + "=" * 70)
    print("2. FORENSIC ANALYSIS: WHY DID THEY FAIL?")
    print("=" * 70)

    df, X, y, feature_cols = load_data()
    if df is None: return

    failures = df[df['Label'] == 0]
    successes = df[df['Label'] == 1]

    print(f"Analyzing {len(failures)} Failures vs {len(successes)} Successes.\n")

    # 1. Compare Means
    mean_diff = failures[feature_cols].mean() - successes[feature_cols].mean()

    # Sort by biggest differences
    print("Top features distinguishing Failures from Successes:")
    print("(Positive value = Higher in Failures, Negative = Higher in Successes)\n")

    sorted_diff = mean_diff.abs().sort_values(ascending=False).head(10)

    for feature in sorted_diff.index:
        diff = mean_diff[feature]
        f_val = failures[feature].mean()
        s_val = successes[feature].mean()
        print(f"{feature:<25}: Diff {diff:+.2f} (Fail Avg: {f_val:.2f} vs Succ Avg: {s_val:.2f})")

    # 2. Look at the specific failure rows
    print("\n--- The Specific Failed Trials ---")

    # Check if these columns exist in original df or need to be reconstructed from features
    # Simplification: Just print the ID and Status from the labeled df
    for idx, row in failures.iterrows():
        print(f"\nID: {row['NCTId']} ({row['OverallStatus']})")
        print(f"  Title: {row['BriefTitle'][:60]}...")
        # Print a few key traits
        print(f"  Enrollment: {row['EnrollmentCount']}")


def main():
    """Run tests."""
    print("\n" + "=" * 70)
    print("ADHD SMALL DATA VALIDATION")
    print("=" * 70)

    best_model = test_loocv_with_oversampling()
    analyze_feature_importance_on_failures()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
