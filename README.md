# House Price Prediction

A regression project comparing Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, KNN, and SVR for predicting house sale prices.

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
2. **Feature engineering** — categorical columns (none currently used) are one-hot encoded, fit only on the training set (`handle_unknown='ignore'` guards against unseen categories at transform time) and applied to the test set with `.transform()`.
3. **Hyperparameter tuning** — `RandomForestRegressor` and `GradientBoostingRegressor` are tuned using `GridSearchCV` (5-fold CV) with `StandardScaler` and the model combined inside a single `sklearn.pipeline.Pipeline`. This ensures scaling is refit independently within each CV fold, so no fold's validation data ever influences the scaling parameters used to transform it. Raw (unscaled) training data is passed into `grid_search.fit()`; the pipeline handles scaling internally, per fold.
4. **Best model extraction** — the tuned model is pulled out of the fitted pipeline (`grid_search.best_estimator_.named_steps['model']`) and its full parameter set is reused via `.get_params()` when building the final model for comparison, ensuring every tuned hyperparameter (including ones like `max_features`) carries over correctly.
5. **Feature scaling for final comparison** — after tuning, `StandardScaler` is fit once on the full training set and applied to the test set, for the final side-by-side model comparison.
6. **Target scaling for SVR** — `SVR` is sensitive to the scale of the target variable as well as the features, so `SalePrice` is separately standardized for the two SVR models only, then predictions are inverse-transformed back to the original dollar scale before computing metrics. Other models use `SalePrice` unscaled, since scaling the target isn't necessary for them.
7. **Evaluation** — models are compared on a held-out test split using MSE, RMSE, and R².

## Models Compared

| Model | Test R² | Train R² |
|---|---|---|
| Linear Regression | 0.9654 | 0.9697 |
| Ridge Regression | 0.9654 | 0.9697 |
| Lasso Regression | 0.9654 | 0.9697 |
| Random Forest | 0.8363 | 0.9746 |
| Gradient Boosting | 0.9350 | 0.9886 |
| KNN (k=5) | 0.8819 | 0.9166 |
| SVR (linear kernel) | 0.9648 | 0.9694 |
| SVR (RBF kernel) | 0.9278 | 0.9745 |

## Key Finding

Linear Regression (and its regularized variants, plus SVR with a linear kernel) clearly outperform every non-linear model tested. This is likely because:

- The dataset is small (300 rows), which limits how well non-linear models (tree ensembles, KNN, RBF-kernel SVR) can learn robust, generalizable patterns.
- The relationship between the chosen features (e.g. `OverallQual`, `GrLivArea`) and `SalePrice` is largely linear, so simpler, linear-shaped models generalize better.
- **SVR (linear kernel) scoring almost identically to plain Linear Regression (0.9648 vs 0.9654) is a strong independent confirmation of this** — a completely different algorithm family, using a different objective (margin-based rather than least-squares), converges on essentially the same conclusion: the underlying relationship is close to linear.
- Random Forest and KNN show the largest train/test gaps (RF: 0.97 vs 0.84; KNN: 0.92 vs 0.88), indicating they're overfitting to local patterns in the small training set rather than capturing generalizable structure.

Ridge and Lasso regression were tested to check whether regularization improved on plain Linear Regression, but all three produced virtually identical scores. This indicates the base Linear Regression model was **not overfitting** to begin with, so there was no variance for regularization to reduce — further confirming that the underlying relationship in this dataset is close to linear.

## Note on Random Forest's score

Random Forest's test R² (0.8363) is lower in this run than in an earlier version (0.8782) that didn't fully apply the tuned `max_features` parameter. This isn't a bug — cross-validation selects hyperparameters based on average performance across folds, which doesn't guarantee identical or improved performance on one specific held-out test set, especially with a model like Random Forest that's sensitive to the small dataset size. The current version correctly reuses every tuned parameter via `.get_params()`, so this number reflects the actual CV-selected configuration rather than a partially-applied one.

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

- Tune `KNeighborsRegressor`'s `n_neighbors` and `SVR`'s `C`/`gamma` via the same leak-free `GridSearchCV` + `Pipeline` pattern already used for Random Forest and Gradient Boosting, rather than using default/fixed values.
- Train on a larger dataset (e.g. the full ~1400-row Ames Housing dataset) to give non-linear models (trees, KNN, RBF-SVR) more data to learn robust patterns from.
- `SalePrice` skewness was checked (0.078) and found to be approximately symmetric, so no log-transform was applied to the target.
- Add residual plots and prediction-vs-actual visualizations for the best model.