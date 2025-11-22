"""
Run the complete ADHD clinical trials prediction pipeline.

This script executes all steps in sequence:
1. Fetch data from ClinicalTrials.gov
2. Prepare and label data
3. Train and evaluate models
"""

import sys
import time


def run_pipeline():
    """Execute the complete pipeline."""
    print("="*70)
    print("ADHD CLINICAL TRIALS SUCCESS PREDICTION PIPELINE")
    print("="*70)
    print()

    start_time = time.time()

    # Step 1: Fetch data
    print("\n" + "="*70)
    print("STEP 1: FETCHING DATA FROM CLINICALTRIALS.GOV")
    print("="*70)
    try:
        from src import fetch_data
        fetch_data.main()
    except Exception as e:
        print(f"ERROR in data fetching: {e}")
        sys.exit(1)

    # Step 2: Prepare data
    print("\n" + "="*70)
    print("STEP 2: PREPARING AND LABELING DATA")
    print("="*70)
    try:
        from src import prepare_data
        prepare_data.main()
    except Exception as e:
        print(f"ERROR in data preparation: {e}")
        sys.exit(1)

    # Step 3: Train models
    print("\n" + "="*70)
    print("STEP 3: TRAINING AND EVALUATING MODELS")
    print("="*70)
    try:
        from src import train_models
        train_models.main()
    except Exception as e:
        print(f"ERROR in model training: {e}")
        sys.exit(1)

    # Summary
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Total execution time: {minutes}m {seconds}s")
    print()
    print("Output files generated:")
    print("  - data/raw/adhd_trials_raw.csv")
    print("  - data/processed/adhd_trials_labeled.csv")
    print("  - data/processed/model_performance.csv")
    print("  - data/processed/roc_curves.png")
    print("  - data/processed/feature_importance_*.png")
    print("  - data/processed/confusion_matrix_*.png")
    print()
    print("Next steps:")
    print("  1. Review the model performance in data/processed/model_performance.csv")
    print("  2. Examine visualizations in data/processed/")
    print("  3. Explore interactively in notebooks/01_eda_and_model.ipynb")
    print("="*70)


if __name__ == "__main__":
    run_pipeline()
