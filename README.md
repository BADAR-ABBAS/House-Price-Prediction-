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

1. **Train/test split first** — the raw dataset is split into train and test sets before any preprocessing happens, to avoid data leakage.
2. **Feature engineering** — categorical columns (none currently used) would be one-hot encoded, fit only on the training set and applied to the test set with `.transform()`.
3. **Feature scaling** — numerical features are standardized with `StandardScaler`, fit only on the training set and applied to the test set with `.transform()`. This ensures the test set is never used to compute the scaling parameters.
4. **Model selection** — `GridSearchCV` (5-fold CV) is used to tune hyperparameters for Random Forest and Gradient Boosting Regressors.
5. **Evaluation** — models are compared on a held-out test split using MSE, RMSE, and R².

## Models Compared

| Model | Test R² | Train R² |
|---|---|---|
| Linear Regression | 0.9532 | 0.9697 |
| Ridge Regression | 0.9527 | 0.9697 |
| Lasso Regression | 0.9532 | 0.9697 |
| Random Forest | 0.8712 | 0.9799 |
| Gradient Boosting | 0.9143 | 0.9890 |

*Note: these scores are slightly lower than an earlier version of this project, where the scaler was fit on the full dataset before splitting. That earlier setup leaked test-set statistics into the training data via the scaler, which mildly inflated the reported scores. The numbers above reflect the corrected split-then-transform pipeline and are a more honest estimate of real-world performance.*

## Key Finding

Linear Regression (and its regularized variants) clearly outperformed both tree-based ensemble methods on this dataset. This is likely because:

- The dataset is small (300 rows), which limits how well tree ensembles can learn robust splits.
- The relationship between the chosen features (e.g. `OverallQual`, `GrLivArea`) and `SalePrice` is largely linear, so a simpler model generalizes better.
- Random Forest showed a noticeable train/test gap (0.98 vs 0.87 R²), indicating overfitting — extensive hyperparameter tuning (including `max_features` and `min_samples_leaf`) did not meaningfully close this gap.

Ridge and Lasso regression were tested to check whether regularization improved on plain Linear Regression, but all three produced virtually identical scores. This indicates the base Linear Regression model was **not overfitting** to begin with, so there was no variance for regularization to reduce — further confirming that the underlying relationship in this dataset is close to linear.

## Known Limitation

`GridSearchCV`'s internal 5-fold cross-validation is currently run on already-scaled training data (scaled once, using statistics from the full training set). This means each CV fold's validation slice was technically part of the data that shaped the scaler used to transform it — a smaller, fold-level version of the leakage issue fixed at the train/test level. It only affects hyperparameter selection, not the final reported test R², since the actual test set is still scaled independently. The proper fix is wrapping `StandardScaler` and each model inside a single `sklearn.pipeline.Pipeline` and passing that pipeline into `GridSearchCV`, so scaling is refit inside each fold automatically.

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
