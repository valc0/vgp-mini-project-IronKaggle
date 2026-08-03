# House Price Regression — King County

A regression project predicting house sale prices from property characteristics, built as an Ironhack Data Science ML mini-project.

## Task

Train a regression model on King County house sales to predict `price` from features like square footage, location, condition, and grade.

## Repository structure
```
.
├── code/
│   └── mini-project-IronKaggle.ipynb
├── dataset/
│   └── king_country_houses_aa.csv
└── README.md
```

## Data
| | Rows | Notes |
|---|---|---|
| Full dataset | 21,613 | 21 features, target = `price`, no duplicates or missing values |

## Pipeline

1. **Cleaning:** Check duplicates/missing values, drop `id`, `date` (converted to `timeline_days`), `zipcode`, and `sqft_above` (multicollinear with `sqft_living`).
2. **EDA:** Correlation heatmaps (raw + post-split leakage audit), distribution histograms.
3. **Encoding:** None needed — all features are already numeric (binary or ordinal).
4. **Baseline Models:** Linear Regression, Ridge, KNN, Random Forest, Gradient Boosting — each scaled and unscaled, scored by R², RMSE, and MAE.
5. **Improvement:**
   - **Data side:** IQR-based outlier removal + a sanity cap on `bedrooms`.
   - **Model side:** XGBoost, plus `RandomizedSearchCV` tuning for Random Forest and XGBoost.
6. **Model Selection:** All baseline + improved models compared on R²/RMSE/MAE; best model chosen automatically.
7. **Feature Importance:** Extracted from the winning model's native importances/coefficients, or permutation importance as a fallback (e.g. for KNN).

## How to Run
Run `mini-project-IronKaggle.ipynb` top to bottom with `king_country_houses_aa.csv` in the `/dataset` directory.

## Deliverables
- [x] Documented notebook (`mini-project-IronKaggle.ipynb`)
- [x] Model comparison (R² / RMSE / MAE)
- [x] Feature importance analysis
