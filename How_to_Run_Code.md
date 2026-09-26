
# How to Run the Crude Oil Forecasting Code

This guide provides step-by-step instructions to set up your environment and run the `code_full.py` or `code_full.ipynb` script for forecasting crude oil prices using VARMAX, XGBoost, and LSTM models.

---

## Required Files

1. `code_full.py` (or `code_full.ipynb`)
2. `CMO-Historical-Data-Monthly.xlsx`

Place both files in the same working directory before running the code.

---

## Step 1: Set Up a Virtual Environment (Recommended)

```bash
# Create a new virtual environment (replace 'venv' with your preferred name)
python -m venv venv

# Activate the environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

---

## Step 2: Install Required Packages

Install dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Step 3: Run the Script

### Option A: Run as Python Script
```bash
python code_full.py
```

### Option B: Run in Jupyter Notebook
```bash
jupyter notebook code_full.ipynb
```

---

## Output

- Plots and forecast charts will be saved in the `plots/` directory.
- Results tables will be written to the `tables/` directory.
- Ensure these folders exist or are created during execution.

---

## Notes

- Ensure that the file `CMO-Historical-Data-Monthly.xlsx` is not open in Excel when the script runs.
- Execution may take several minutes depending on system specifications and model complexity.
