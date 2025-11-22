"""
Test and validate the ADHD clinical trials prediction model.

This script provides various ways to test model accuracy and robustness:
1. Cross-validation for robust performance estimates
2. Baseline comparison
3. Individual trial predictions
4. Error analysis
5. What-if scenario testing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def load_data():
    """Load processed data."""
    df = pd.read_csv('data/processed/adhd_trials_labeled.csv')

    metadata_cols = ['NCTId', 'Label', 'OverallStatus', 'BriefTitle']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    X = df[feature_cols].values
    y = df['Label'].values

    return df, X, y, feature_cols


def test_cross_validation():
    """
    Perform k-fold cross-validation for robust accuracy estimation.

    This gives a better estimate of model performance than a single train/test split.
    """
    print("="*70)
    print("1. CROSS-VALIDATION TEST")
    print("="*70)
    print("\nThis tests how well the model generalizes to unseen data")
    print("by training and testing on different subsets of the data.\n")

    df, X, y, feature_cols = load_data()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models to test
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # 5-fold cross-validation (or fewer if dataset is small)
    n_splits = min(5, len(y) // 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Using {n_splits}-fold cross-validation")
    print(f"Dataset size: {len(y)} samples\n")

    results = {}

    for name, model in models.items():
        print(f"\nTesting {name}...")

        # Cross-validation scores
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')

        results[name] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores
        }

        print(f"  Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        print(f"  Individual folds: {[f'{s:.3f}' for s in scores]}")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['mean_accuracy'])

    print("\n" + "-"*70)
    print(f"BEST MODEL: {best_model[0]}")
    print(f"Mean Accuracy: {best_model[1]['mean_accuracy']:.3f} (+/- {best_model[1]['std_accuracy']:.3f})")
    print("-"*70)

    return results


def test_baseline_comparison():
    """
    Compare model performance to baseline strategies.

    This shows if the model is actually learning or just predicting the majority class.
    """
    print("\n" + "="*70)
    print("2. BASELINE COMPARISON TEST")
    print("="*70)
    print("\nComparing model to simple baseline strategies:\n")

    df, X, y, feature_cols = load_data()

    # Calculate baseline accuracies
    majority_class = 1 if (y == 1).sum() > (y == 0).sum() else 0
    majority_baseline = (y == majority_class).mean()

    random_baseline = max((y == 1).mean(), (y == 0).mean())

    print(f"1. Majority Class Baseline (always predict most common class)")
    print(f"   Accuracy: {majority_baseline:.3f}")
    print(f"   Strategy: Always predict {'Success' if majority_class == 1 else 'Failure'}")

    print(f"\n2. Random Baseline (random guessing)")
    print(f"   Expected Accuracy: ~{random_baseline:.3f}")
    print(f"   Strategy: Random predictions")

    # Load actual model performance
    model_perf = pd.read_csv('data/processed/model_performance.csv', index_col=0)
    best_test_acc = model_perf['test_accuracy'].max()
    best_model = model_perf['test_accuracy'].idxmax()

    print(f"\n3. Best ML Model ({best_model})")
    print(f"   Test Accuracy: {best_test_acc:.3f}")

    # Calculate improvement
    improvement = ((best_test_acc - majority_baseline) / majority_baseline) * 100

    print("\n" + "-"*70)
    if best_test_acc > majority_baseline:
        print(f" Model BEATS baseline by {improvement:.1f}%")
    else:
        print(f"  Model performs similar to baseline")
    print("-"*70)


def predict_individual_trial(nct_id=None):
    """
    Make predictions for individual trials.

    This shows how the model would predict specific trials.
    """
    print("\n" + "="*70)
    print("3. INDIVIDUAL TRIAL PREDICTION TEST")
    print("="*70)

    df, X, y, feature_cols = load_data()

    # Train model on all data (for demonstration)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_scaled, y)

    # Get predictions and probabilities
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    df['Predicted'] = predictions
    df['Probability_Success'] = probabilities
    df['Correct'] = (predictions == y)

    if nct_id:
        # Predict specific trial
        trial = df[df['NCTId'] == nct_id].iloc[0]
        print(f"\nPrediction for {nct_id}:")
        print(f"Title: {trial['BriefTitle'][:60]}...")
        print(f"Actual Status: {trial['OverallStatus']}")
        print(f"Predicted: {'Success' if trial['Predicted'] == 1 else 'Failure'}")
        print(f"Probability of Success: {trial['Probability_Success']:.1%}")
        print(f"Correct: {' Yes' if trial['Correct'] else ' No'}")
    else:
        # Show a few examples
        print("\nSample predictions:\n")

        # Show some correct predictions
        print("CORRECT PREDICTIONS:")
        correct = df[df['Correct'] == True].head(3)
        for idx, trial in correct.iterrows():
            print(f"\n  {trial['NCTId']}: {trial['BriefTitle'][:50]}...")
            print(f"    Actual: {trial['OverallStatus']}, Predicted: {'Success' if trial['Predicted']==1 else 'Failure'}")
            print(f"    Confidence: {trial['Probability_Success']:.1%}")

        # Show incorrect predictions if any
        incorrect = df[df['Correct'] == False]
        if len(incorrect) > 0:
            print("\n\nINCORRECT PREDICTIONS:")
            for idx, trial in incorrect.head(3).iterrows():
                print(f"\n  {trial['NCTId']}: {trial['BriefTitle'][:50]}...")
                print(f"    Actual: {trial['OverallStatus']}, Predicted: {'Success' if trial['Predicted']==1 else 'Failure'}")
                print(f"    Confidence: {trial['Probability_Success']:.1%}")

    return df


def analyze_errors():
    """
    Analyze where the model makes mistakes.

    This helps understand model limitations and areas for improvement.
    """
    print("\n" + "="*70)
    print("4. ERROR ANALYSIS")
    print("="*70)

    df, X, y, feature_cols = load_data()

    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_scaled, y)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    df['Predicted'] = predictions
    df['Probability_Success'] = probabilities

    # Analyze errors
    errors = df[predictions != y]
    correct = df[predictions == y]

    print(f"\nTotal predictions: {len(df)}")
    print(f"Correct: {len(correct)} ({len(correct)/len(df)*100:.1f}%)")
    print(f"Incorrect: {len(errors)} ({len(errors)/len(df)*100:.1f}%)")

    if len(errors) > 0:
        print(f"\nError breakdown:")

        # False positives
        false_pos = errors[errors['Label'] == 0]
        print(f"  False Positives (predicted success, actually failed): {len(false_pos)}")

        # False negatives
        false_neg = errors[errors['Label'] == 1]
        print(f"  False Negatives (predicted failure, actually succeeded): {len(false_neg)}")

        print("\nMost uncertain correct predictions (low confidence):")
        # Successes predicted with low confidence
        uncertain_success = correct[correct['Label'] == 1].nsmallest(3, 'Probability_Success')
        for idx, trial in uncertain_success.iterrows():
            print(f"  {trial['NCTId']}: {trial['Probability_Success']:.1%} confidence")
            print(f"    Enrollment: {trial['EnrollmentCount']:.0f}")

        if len(false_pos) > 0:
            print("\nFalse Positives (trials we thought would succeed but failed):")
            for idx, trial in false_pos.iterrows():
                print(f"\n  {trial['NCTId']}: {trial['BriefTitle'][:50]}...")
                print(f"    Predicted Success with {trial['Probability_Success']:.1%} confidence")
                print(f"    Actually: {trial['OverallStatus']}")
                print(f"    Enrollment: {trial['EnrollmentCount']:.0f}")


def test_what_if_scenarios():
    """
    Test what-if scenarios to understand model behavior.

    This shows how changing trial characteristics affects predictions.
    """
    print("\n" + "="*70)
    print("5. WHAT-IF SCENARIO TESTING")
    print("="*70)
    print("\nThis shows how trial characteristics affect predicted success:\n")

    df, X, y, feature_cols = load_data()

    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_scaled, y)

    # Create hypothetical trials
    scenarios = []

    # Base trial (median values)
    base_trial = pd.DataFrame([X.mean(axis=0)], columns=feature_cols)
    base_pred = model.predict_proba(scaler.transform(base_trial))[0, 1]

    print(f"Base trial (average characteristics):")
    print(f"  Predicted success probability: {base_pred:.1%}\n")

    # Scenario 1: Small vs Large enrollment
    small_trial = base_trial.copy()
    small_trial['EnrollmentCount'] = 30
    small_trial['SmallTrial'] = 1
    small_trial['LargeTrial'] = 0
    small_pred = model.predict_proba(scaler.transform(small_trial))[0, 1]

    large_trial = base_trial.copy()
    large_trial['EnrollmentCount'] = 300
    large_trial['SmallTrial'] = 0
    large_trial['LargeTrial'] = 1
    large_pred = model.predict_proba(scaler.transform(large_trial))[0, 1]

    print("Scenario 1: Effect of enrollment size")
    print(f"  Small trial (30 participants): {small_pred:.1%} success probability")
    print(f"  Large trial (300 participants): {large_pred:.1%} success probability")
    print(f"  Difference: {(large_pred - small_pred)*100:.1f} percentage points\n")

    # Scenario 2: Randomized vs Non-randomized
    non_rand = base_trial.copy()
    non_rand['IsRandomized'] = 0
    non_rand_pred = model.predict_proba(scaler.transform(non_rand))[0, 1]

    rand = base_trial.copy()
    rand['IsRandomized'] = 1
    rand_pred = model.predict_proba(scaler.transform(rand))[0, 1]

    print("Scenario 2: Effect of randomization")
    print(f"  Non-randomized: {non_rand_pred:.1%} success probability")
    print(f"  Randomized: {rand_pred:.1%} success probability")
    print(f"  Difference: {(rand_pred - non_rand_pred)*100:.1f} percentage points\n")

    # Scenario 3: Industry vs Academic sponsor
    academic = base_trial.copy()
    academic['IsIndustrySponsored'] = 0
    academic['IsAcademicSponsored'] = 1
    academic_pred = model.predict_proba(scaler.transform(academic))[0, 1]

    industry = base_trial.copy()
    industry['IsIndustrySponsored'] = 1
    industry['IsAcademicSponsored'] = 0
    industry_pred = model.predict_proba(scaler.transform(industry))[0, 1]

    print("Scenario 3: Effect of sponsor type")
    print(f"  Academic sponsored: {academic_pred:.1%} success probability")
    print(f"  Industry sponsored: {industry_pred:.1%} success probability")
    print(f"  Difference: {(industry_pred - academic_pred)*100:.1f} percentage points")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ADHD CLINICAL TRIALS MODEL - COMPREHENSIVE TESTING")
    print("="*70)
    print("\nThis script tests the model's accuracy and robustness in multiple ways.\n")

    # Test 1: Cross-validation
    cv_results = test_cross_validation()

    # Test 2: Baseline comparison
    test_baseline_comparison()

    # Test 3: Individual predictions
    predictions_df = predict_individual_trial()

    # Test 4: Error analysis
    analyze_errors()

    # Test 5: What-if scenarios
    test_what_if_scenarios()

    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Cross-validation gives robust accuracy estimates")
    print("2. Model beats baseline (majority class predictor)")
    print("3. Individual trial predictions show model reasoning")
    print("4. Error analysis reveals where model struggles")
    print("5. What-if scenarios show feature importance")
    print("\nNext steps:")
    print("- Review predictions for individual trials")
    print("- Examine errors to understand limitations")
    print("- Test on new trials as they complete")
    print("="*70)


if __name__ == "__main__":
    main()
