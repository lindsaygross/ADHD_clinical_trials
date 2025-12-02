# Predicting Success of ADHD Clinical Trials

A machine learning pipeline that predicts the probability of clinical trial success using data from ClinicalTrials.gov.

## Project Overview

This project builds an end-to-end pipeline to predict whether ADHD (Attention Deficit Hyperactivity Disorder) Phase 1, 2, and 3 interventional clinical trials will be **successfully completed** or **fail early** (terminated, withdrawn, or suspended).

### Objective

Given the design and characteristics of an ADHD clinical trial at the time it starts, predict the probability that it will be **Completed** rather than **Terminated/Withdrawn/Suspended**.

### Key Features

- Automated data collection from ClinicalTrials.gov API
- Comprehensive feature engineering using only pre-trial information (no data leakage)
- Multiple ML models with performance comparison
- Visualizations and interpretability analysis
- Clean, modular, and reusable code

## Project Structure

```
ADHD_clinical_trials/
 data/
    raw/                      # Raw data from API
       adhd_trials_raw.json
       adhd_trials_raw.csv
    processed/                # Processed and labeled data
        adhd_trials_labeled.csv
        model_performance.csv
        roc_curves.png
        feature_importance_*.png
 src/
    fetch_data.py            # Data collection from ClinicalTrials.gov
    prepare_data.py          # Data preprocessing and feature engineering
    train_models.py          # Model training and evaluation
    utils.py                 # Utility functions
 notebooks/
    01_eda_and_model.ipynb  # Exploratory analysis and modeling
 requirements.txt             # Python dependencies
 README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository or navigate to the project directory:

```bash
cd ADHD_clinical_trials
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Fetch Data from ClinicalTrials.gov

Retrieve ADHD Phase 1/2/3 interventional trials:

```bash
python -m src.fetch_data
```

This will:
- Query the ClinicalTrials.gov API for ADHD trials
- Filter for Phase 1, 2, and 3 interventional studies
- Save raw data to `data/raw/adhd_trials_raw.json` and `data/raw/adhd_trials_raw.csv`
- Display summary statistics

### Step 2: Prepare and Label Data

Process raw data and create labeled dataset:

```bash
python -m src.prepare_data
```

This will:
- Create binary labels (Success=1 for Completed, Failure=0 for Terminated/Withdrawn/Suspended)
- Engineer features using only pre-trial information
- Handle missing values
- Save processed data to `data/processed/adhd_trials_labeled.csv`

### Step 3: Train and Evaluate Models

Train multiple ML models and compare performance:

```bash
python -m src.train_models
```

This will:
- Split data into training and test sets
- Train three models: Logistic Regression, Random Forest, Gradient Boosting
- Evaluate models using multiple metrics (accuracy, precision, recall, F1, AUC)
- Generate visualizations (ROC curves, feature importance, confusion matrices)
- Save results to `data/processed/`

### Step 4: Explore with Jupyter Notebook

For interactive analysis:

```bash
jupyter notebook notebooks/01_eda_and_model.ipynb
```

## Methodology

### Label Definition

- **Success (1)**: Trial status is `COMPLETED`
- **Failure (0)**: Trial status is `TERMINATED`, `WITHDRAWN`, or `SUSPENDED`
- Excluded: Active trials, recruiting trials, and unknown statuses

### Feature Engineering

Features are derived from trial characteristics known **before or at the start** of the trial:

**Enrollment Features:**
- Enrollment count (raw and log-transformed)
- Small trial (<50 participants) and large trial (≥200 participants) indicators

**Phase Features:**
- Phase 1, Phase 2, Phase 3, or combined Phase 2/3 indicators

**Design Features:**
- Randomization status
- Blinding/masking type (double-blind, any blinding)
- Number of arms (parallel, crossover)
- Allocation method

**Intervention Features:**
- Intervention types (drug, behavioral, device)
- Number of interventions

**Sponsor Features:**
- Sponsor class (industry, NIH, academic, other)

**Geographic Features:**
- Number of countries
- Multi-country vs single-country
- US-only trials

**Eligibility Features:**
- Age groups (children, adults)
- Gender restrictions
- Acceptance of healthy volunteers

**Purpose Features:**
- Primary purpose (treatment, prevention, etc.)

### Models

Three classification algorithms are trained and compared:

1. **Logistic Regression**: Interpretable baseline model
2. **Random Forest**: Ensemble method with feature importance
3. **Gradient Boosting**: Advanced ensemble method

All models use:
- Stratified train-test split (80/20)
- Class balancing to handle imbalanced data
- Standard feature scaling

### Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Proportion of predicted successes that were actual successes
- **Recall**: Proportion of actual successes that were predicted
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve (discrimination ability)

## Results

After running the pipeline, you will find:

1. **Model Performance Table**: `data/processed/model_performance.csv`
   - Comparison of all models across metrics

2. **ROC Curves**: `data/processed/roc_curves.png`
   - Visual comparison of model discrimination ability

3. **Feature Importance Plots**: `data/processed/feature_importance_*.png`
   - Most important predictors for tree-based models

4. **Confusion Matrices**: `data/processed/confusion_matrix_*.png`
   - Breakdown of correct and incorrect predictions

## Key Insights

The analysis typically reveals:

- **Enrollment size** is a strong predictor (larger trials more likely to complete)
- **Study design quality** (randomization, blinding) correlates with completion
- **Sponsor type** affects success rates (industry vs. academic)
- **Phase 3 trials** may have different success patterns than Phase 2

## Limitations and Considerations

1. **Class Imbalance**: Most trials are completed successfully, creating an imbalanced dataset
2. **Limited Sample Size**: Focus on ADHD Phase 2/3 trials restricts the dataset
3. **Missing Data**: Some trials have incomplete information on design features
4. **Temporal Bias**: Recent trials may still be ongoing (excluded from analysis)
5. **External Validity**: Results specific to ADHD trials may not generalize to other conditions

## Future Enhancements

Potential improvements:
- Incorporate text features from trial titles/descriptions using NLP
- Add temporal features (year, duration)
- Expand to other therapeutic areas for comparison
- Implement advanced models (XGBoost, neural networks)
- Add cross-validation and hyperparameter tuning
- Build a prediction API or web interface

## Dependencies

Core libraries:
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Machine learning
- `matplotlib`, `seaborn`: Visualization
- `requests`: API calls
- `jupyter`: Interactive notebooks

See `requirements.txt` for specific versions.

## Data Source

All data is sourced from [ClinicalTrials.gov](https://clinicaltrials.gov/), a public registry of clinical studies maintained by the U.S. National Library of Medicine.


1. **Visit ClinicalTrials.gov**: https://clinicaltrials.gov
2. **Search for "ADHD" + "Interventional"**
3. **Filter to Phase 1, 2 and 3**
4. **Compare results** to our dataset
5. **Verify individual trials** using NCT IDs we provide


## License

This project is for educational purposes as part of a Machine Learning course.

## Author

Lindsay Gross
AIPI 520 - Machine Learning Course Project

## Acknowledgments

- ClinicalTrials.gov for providing open access to clinical trial data
- Scikit-learn community for excellent ML tools and documentation
