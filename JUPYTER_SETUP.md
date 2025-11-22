# Jupyter Notebook Setup Guide

## Problem: ModuleNotFoundError

If you see `ModuleNotFoundError: No module named 'pandas'` when running the notebook, it means Jupyter is not using the correct Python kernel.

## Solution

### Step 1: Make Sure Packages are Installed

Activate your virtual environment and ensure all packages are installed:

```bash
cd "/Users/lindsaygross/ME AIPI Code/AIPI520/aipi520_repo/ADHDTrials/ADHD_clinical_trials"
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Launch Jupyter from Virtual Environment

Always launch Jupyter from within the activated virtual environment:

```bash
# Make sure venv is activated (you should see (venv) in your prompt)
source venv/bin/activate

# Launch Jupyter
jupyter notebook
```

### Step 3: Select the Correct Kernel

When the notebook opens:

1. Click **Kernel** in the menu bar
2. Select **Change Kernel**
3. Choose **"Python (ADHD Trials)"** from the dropdown

This kernel was specifically created for this project and has access to all installed packages.

## Alternative: Using JupyterLab

If you prefer JupyterLab over Jupyter Notebook:

```bash
source venv/bin/activate
jupyter lab
```

Then select the "Python (ADHD Trials)" kernel.

## Troubleshooting

### Issue: "Python (ADHD Trials)" kernel not appearing

Re-register the kernel:

```bash
source venv/bin/activate
python -m ipykernel install --user --name=adhd_trials --display-name="Python (ADHD Trials)"
```

### Issue: Kernel keeps dying or restarting

Check that all packages are installed:

```bash
source venv/bin/activate
pip list | grep -E "pandas|numpy|scikit-learn|matplotlib"
```

If any are missing:

```bash
pip install -r requirements.txt
```

### Issue: Still getting ModuleNotFoundError

1. Restart Jupyter completely
2. Close all notebooks
3. Shut down the Jupyter server (Ctrl+C in terminal)
4. Relaunch from activated venv:

```bash
source venv/bin/activate
jupyter notebook
```

## Quick Reference

### Every Time You Want to Use the Notebook:

```bash
# 1. Navigate to project
cd "/Users/lindsaygross/ME AIPI Code/AIPI520/aipi520_repo/ADHDTrials/ADHD_clinical_trials"

# 2. Activate virtual environment
source venv/bin/activate

# 3. Launch Jupyter
jupyter notebook notebooks/01_eda_and_model.ipynb

# 4. Select kernel: Kernel > Change Kernel > Python (ADHD Trials)
```

## Verifying the Setup

Run this in the first cell of your notebook:

```python
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

import pandas
import numpy
import sklearn
print("\nAll packages imported successfully!")
```

You should see the path pointing to your venv directory.

## Using VS Code Instead

If you're using VS Code to run Jupyter notebooks:

1. Open the notebook in VS Code
2. Click the kernel selector in the top right
3. Choose **"Python (ADHD Trials)"** or the path to `venv/bin/python`
4. Run cells normally

## Summary

The key is to **always launch Jupyter from within the activated virtual environment** and **select the correct kernel** that has access to all your installed packages.
