# Beijing PM2.5 Forecasting

Predict 24-hour-ahead PM2.5 at Beijing air-quality stations from concurrent pollutants, weather, and recent station/city history. The final model is an XGBoost regressor with a temporal holdout for tuning.

---

## Objective

Estimate `PM2.5_24h_ahead` for each test observation and write a Kaggle-style submission file.

| Item | Detail |
| --- | --- |
| Target | `PM2.5_24h_ahead` |
| ID | `observation_id` |
| Metric | RMSE |
| Output | `project/volume/data/processed/submission.csv` |

---

## Data

Place competition files in `project/volume/data/raw/` (not modified by the pipeline):

| File | Role |
| --- | --- |
| `STAT_380_Train_P3.csv` | Labeled training rows |
| `STAT_380_Test_P3.csv` | Unlabeled test rows |
| `sample_submission.csv` | Required `observation_id` order for submission |

Raw fields used as predictors include station pollutants (`PM10`, `SO2`, `NO2`, `CO`, `O3`), weather (`TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, `wd`), and calendar fields derived from `forecast_origin`. The label `PM2.5_24h_ahead` is never used as a feature.

---

## Approach

### Features (`project/src/features/build_features.R`)

1. Stack train and test, sort by `station` and `forecast_origin`.
2. Citywide means at the same hour (`city_*`) across all stations.
3. 24h / 48h lags for selected pollutants, weather, and city means.
4. Differences: `PM10_change_24`, `PM10_vs_city`.
5. Cyclical time encodings for hour and month, plus weekday.
6. One-hot station and wind direction (with a missing-wind indicator).
7. Write interim feature tables:
   - `project/volume/data/interim/train_features.csv`
   - `project/volume/data/interim/test_features.csv`

### Model (`project/src/models/train_model.R`)

1. Temporal holdout: fit on `forecast_origin < 2015-03-01`, validate on later hours.
2. Train XGBoost (`reg:squarederror`, RMSE) with early stopping to choose tree count.
3. Refit on all training rows with that tree count.
4. Predict test rows, clip at 0, join to `sample_submission` order.
5. Save:
   - `project/volume/data/processed/submission.csv`
   - `project/volume/models/pm25_xgb.model`

---

## Reproduce

From the repository root:

```bash
# Install R packages (data.table, glmnet, xgboost)
Rscript project/required/requirements.R

# Run the declared final workflow
bash project/entry_point.sh
```

`project/entry_point.sh` runs, in order:

1. `project/src/features/build_features.R`
2. `project/src/models/train_model.R`

Optional data profiling (not part of the submission path):

```bash
Rscript project/src/data/profile_data.R
```

---

## Project layout

```text
kaggle-beijing-pm2.5-forcasting/
├── README.md
├── AGENTS.md
└── project/
    ├── PROJECT_CONTEXT.md
    ├── entry_point.sh
    ├── required/
    │   ├── requirements.R
    │   └── requirements.txt
    ├── src/
    │   ├── data/profile_data.R
    │   ├── features/build_features.R
    │   └── models/train_model.R
    └── volume/
        ├── data/
        │   ├── raw/          # competition CSVs (immutable)
        │   ├── interim/      # engineered features
        │   ├── processed/    # submission.csv
        │   └── profiles/     # compact data profiles
        └── models/           # saved xgboost model
```

Paths in scripts are relative to the repository root (for example `project/volume/data/raw/...`). Do not use machine-specific absolute paths.

---

## Notes

- Raw data stay under `project/volume/data/raw/` and are never overwritten.
- Holdout is time-based so later hours cannot leak into tuning.
- Random seed is fixed at `380` for reproducibility.
- See `project/PROJECT_CONTEXT.md` for the concise problem statement used by collaborators and AI tools.
