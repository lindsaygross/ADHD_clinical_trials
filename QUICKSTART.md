# Quick Start Guide

This guide will help you get up and running with the ADHD Clinical Trials prediction pipeline in just a few minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for API calls)

## Installation Steps

### 1. Navigate to Project Directory

```bash
cd ADHD_clinical_trials
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Option 1: Run Complete Pipeline (Recommended for First Time)

Execute all steps automatically:

```bash
python run_pipeline.py
```

This will:
1. Fetch ADHD trial data from ClinicalTrials.gov (~2-3 minutes)
2. Process and label the data (~30 seconds)
3. Train and evaluate models (~1-2 minutes)
4. Generate all visualizations

**Total time: ~5 minutes**

### Option 2: Run Steps Individually

If you prefer to run each step separately:

**Step 1: Fetch Data**
```bash
python -m src.fetch_data
```

**Step 2: Prepare Data**
```bash
python -m src.prepare_data
```

**Step 3: Train Models**
```bash
python -m src.train_models
```

### Option 3: Interactive Analysis

For interactive exploration in Jupyter:

```bash
jupyter notebook notebooks/01_eda_and_model.ipynb
```

## Expected Outputs

After running the pipeline, you'll find:

### Data Files
- `data/raw/adhd_trials_raw.csv` - Raw trial data from API
- `data/raw/adhd_trials_raw.json` - Raw JSON response
- `data/processed/adhd_trials_labeled.csv` - Processed dataset with features

### Results
- `data/processed/model_performance.csv` - Performance metrics for all models
- `data/processed/feature_names.json` - List of features used

### Visualizations
- `data/processed/roc_curves.png` - ROC curves comparing models
- `data/processed/feature_importance_random_forest.png` - Top features (Random Forest)
- `data/processed/feature_importance_gradient_boosting.png` - Top features (Gradient Boosting)
- `data/processed/confusion_matrix_[model_name].png` - Confusion matrices

## Quick Data Overview

Expected dataset size:
- **Total trials**: ~200-400 ADHD Phase 2/3 trials
- **Labeled trials**: ~100-200 (completed or failed)
- **Features**: 34 engineered features
- **Success rate**: ~70-80% (most trials complete successfully)

## Troubleshooting

### Issue: API Rate Limiting

If you get rate-limited by ClinicalTrials.gov:
- The script includes built-in delays between requests
- If needed, increase `delay_seconds` in [fetch_data.py](src/fetch_data.py):110

### Issue: Missing Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Issue: Insufficient Data

If too few trials are fetched:
- Check internet connection
- Increase `max_results` in [fetch_data.py](src/fetch_data.py):110
- Verify ClinicalTrials.gov API is accessible

### Issue: Module Not Found

Make sure you're in the project root directory:
```bash
cd ADHD_clinical_trials
python -m src.fetch_data  # Note the -m flag
```

## Next Steps

After running the pipeline:

1. **Review Model Performance**
   - Open `data/processed/model_performance.csv`
   - Compare accuracy, precision, recall, F1, and AUC across models

2. **Examine Visualizations**
   - Check ROC curves to see model discrimination ability
   - Review feature importance to understand key predictors
   - Analyze confusion matrices for error patterns

3. **Interactive Analysis**
   - Open the Jupyter notebook for deeper exploration
   - Modify features or model parameters
   - Perform additional analyses

4. **Customize for Your Needs**
   - Adjust features in [prepare_data.py](src/prepare_data.py)
   - Add new models in [train_models.py](src/train_models.py)
   - Create custom visualizations using [utils.py](src/utils.py)

## Project Structure

```
ADHD_clinical_trials/
 data/                    # Data directory
    raw/                # Raw API data
    processed/          # Processed data and results
 src/                     # Source code
    fetch_data.py       # API data fetching
    prepare_data.py     # Data preprocessing
    train_models.py     # Model training
    utils.py            # Utility functions
 notebooks/               # Jupyter notebooks
    01_eda_and_model.ipynb
 requirements.txt         # Dependencies
 run_pipeline.py         # Main pipeline script
 README.md               # Full documentation
 QUICKSTART.md           # This file
```

## Getting Help

- **Full Documentation**: See [README.md](README.md)
- **Code Comments**: All modules have detailed docstrings
- **Issues**: Report problems or ask questions in the repository

## Tips for Success

1. **First Run**: Use `run_pipeline.py` to ensure everything works
2. **Explore**: Use the Jupyter notebook for interactive analysis
3. **Customize**: Modify features or models based on your research questions
4. **Document**: Keep notes on experiments and findings
5. **Version Control**: Commit changes regularly if using git

Happy analyzing!
