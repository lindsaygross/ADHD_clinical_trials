# Setup Guide

## Virtual Environment Setup

A virtual environment (`venv`) has been created for this project. Follow these steps to activate it and install dependencies.

### Activate the Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

When activated, you'll see `(venv)` at the beginning of your command prompt:
```
(venv) user@computer:~/ADHD_clinical_trials$
```

### Install Dependencies

Once the virtual environment is activated, install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- pandas (data manipulation)
- numpy (numerical operations)
- scikit-learn (machine learning)
- matplotlib (plotting)
- seaborn (visualization)
- requests (API calls)
- jupyter (notebooks)

### Verify Installation

Check that everything is installed correctly:

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, requests; print('All packages installed successfully!')"
```

### Deactivate Virtual Environment

When you're done working, deactivate the virtual environment:

```bash
deactivate
```

---

## Quick Start

### 1. Activate venv and install dependencies

```bash
# Navigate to project directory
cd ADHD_clinical_trials

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the complete pipeline

```bash
python run_pipeline.py
```

Or run steps individually:

```bash
# Step 1: Fetch data
python -m src.fetch_data

# Step 2: Prepare data
python -m src.prepare_data

# Step 3: Train models
python -m src.train_models
```

### 3. Test the model

```bash
python test_model.py
```

### 4. Explore in Jupyter

```bash
jupyter notebook notebooks/01_eda_and_model.ipynb
```

---

## Troubleshooting

### Issue: "command not found: python"

Try using `python3` instead:
```bash
python3 -m venv venv
```

### Issue: Permission denied

On macOS/Linux, you may need to make scripts executable:
```bash
chmod +x run_pipeline.py
```

### Issue: pip install fails

Upgrade pip first:
```bash
pip install --upgrade pip
```

Then try installing requirements again:
```bash
pip install -r requirements.txt
```

### Issue: Module not found when running scripts

Make sure:
1. Virtual environment is activated (`source venv/bin/activate`)
2. You're in the project root directory
3. All dependencies are installed (`pip install -r requirements.txt`)

---

## Working with the Virtual Environment

### Why use a virtual environment?

- **Isolation**: Keeps project dependencies separate from system Python
- **Reproducibility**: Ensures everyone uses the same package versions
- **No conflicts**: Prevents package version conflicts with other projects

### Best practices

1. **Always activate venv** before working on the project
2. **Install packages within venv** using `pip install`
3. **Update requirements.txt** if you add new packages:
   ```bash
   pip freeze > requirements.txt
   ```
4. **Deactivate when done** to return to system Python

### Check what's installed

List all installed packages:
```bash
pip list
```

Check specific package version:
```bash
pip show pandas
```

---

## VS Code Integration

If using VS Code, select the virtual environment as your Python interpreter:

1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `./venv/bin/python`

This ensures VS Code uses the correct Python environment.

---

## PyCharm Integration

If using PyCharm:

1. Go to `Settings/Preferences > Project > Python Interpreter`
2. Click the gear icon > `Add`
3. Select `Existing environment`
4. Navigate to `ADHD_clinical_trials/venv/bin/python`
5. Click `OK`

---

## Summary of Commands

```bash
# Activate venv (macOS/Linux)
source venv/bin/activate

# Activate venv (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py

# Test model
python test_model.py

# Deactivate venv
deactivate
```

---

## Next Steps

Once setup is complete:

1.  Virtual environment created and activated
2.  Dependencies installed
3.  Run the pipeline: `python run_pipeline.py`
4.  Review results in `data/processed/`
5.  Explore notebook: `jupyter notebook notebooks/01_eda_and_model.ipynb`

Happy coding! 
