# House Price Prediction

A small regression project comparing Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting models for predicting house sale prices.

## Dataset

- File: `house_prices_prediction.csv`
- Rows: 300
- Features used:
  - `OverallQual`
  - `GrLivArea`
  - `GarageCars`
  - `TotalBsmtSF`
  - `YearBuilt`
  - `FullBath`
- Target: `SalePrice`

## Approach

1. **Feature scaling** — numerical features are standardized with `StandardScaler`.
2. **Model selection** — `GridSearchCV` (5-fold CV) is used to tune hyperparameters for Random Forest and Gradient Boosting Regressors.
3. **Evaluation** — models are compared on a held-out test split using MSE, RMSE, and R².

## Models Compared

| Model | Test R² | Train R² |
|---|---|---|
| Linear Regression | 0.9654 | 0.9697 |
| Ridge Regression | 0.9654 | 0.9697 |
| Lasso Regression | 0.9654 | 0.9697 |
| Random Forest | 0.8793 | 0.9799 |
| Gradient Boosting | 0.9397 | 0.9890 |

## Key Finding

Linear Regression (and its regularized variants) clearly outperformed both tree-based ensemble methods on this dataset. This is likely because:

- The dataset is small (300 rows), which limits how well tree ensembles can learn robust splits.
- The relationship between the chosen features (e.g. `OverallQual`, `GrLivArea`) and `SalePrice` is largely linear, so a simpler model generalizes better.
- Random Forest showed a noticeable train/test gap (0.98 vs 0.88 R²), indicating overfitting — extensive hyperparameter tuning (including `max_features` and `min_samples_leaf`) did not meaningfully close this gap.

Ridge and Lasso regression were tested to check whether regularization improved on plain Linear Regression, but all three produced virtually identical scores (R² within 0.0001 of each other). This indicates the base Linear Regression model was **not overfitting** to begin with, so there was no variance for regularization to reduce — further confirming that the underlying relationship in this dataset is close to linear.

## How to Run

```bash
python main.py
```

## Requirements

```
pandas
matplotlib
seaborn
scikit-learn
```

## Possible Next Steps

- Train on a larger dataset (e.g. the full ~1400-row Ames Housing dataset) to give tree ensembles more data to work with.
- `SalePrice` skewness was checked (0.078) and found to be approximately symmetric, so no log-transform was applied.
- Add residual plots and prediction-vs-actual visualizations for the best model.
