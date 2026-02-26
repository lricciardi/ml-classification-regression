# Machine Learning: Classification & Regression

Implemented regression and classification algorithms with data preprocessing and evaluation in Python.

## Overview

This project demonstrates an end-to-end ML workflow:

- **Data preprocessing**: train/test split, feature scaling (StandardScaler), stratified splits for classification
- **Classification** (Iris dataset): Logistic Regression, Random Forest, SVM (RBF)
- **Regression** (Diabetes dataset): Linear Regression, Ridge, Random Forest Regressor
- **Evaluation**: accuracy, F1, precision, recall, R², MSE, MAE, and cross-validation

## Project structure

```
ml-classification-regression/
├── src/
│   ├── __init__.py
│   ├── train.py    # Main pipeline: load data, train, evaluate
│   ├── model.py    # Model definitions (classifiers & regressors)
│   └── utils.py    # Preprocessing, split, evaluation metrics
├── data/           # Optional — place sample or custom data here
├── notebooks/      # Optional — Jupyter notebooks
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

- Python 3.10+
- numpy
- scikit-learn

## Setup

```bash
cd ml-classification-regression
pip install -r requirements.txt
```

## Run

From the **project root** (`ml-classification-regression/`):

```bash
python -m src.train
```

## Output

- **Classification**: Test accuracy and F1 per model, 5-fold CV F1, and a full classification report for the best model (Iris).
- **Regression**: Test R², MSE, MAE, and 5-fold CV R² for each regressor (Diabetes).

## Skills demonstrated

- Data preprocessing and scaling
- Multiple classification and regression algorithms
- Proper train/test split and cross-validation
- Metric selection (F1 for classification, R²/ MSE/ MAE for regression)
- Clean, modular layout (train / model / utils) with type hints and docstrings
