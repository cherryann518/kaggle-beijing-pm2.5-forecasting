# Project Context

## Objective

Predict 24-hour-ahead PM2.5 (`PM2.5_24h_ahead`) at Beijing air-quality stations from concurrent pollutants, weather, and recent station/city history.

## Data Sources

- Training data: `project/volume/data/raw/STAT_380_Train_P3.csv`
- Test data: `project/volume/data/raw/STAT_380_Test_P3.csv`
- Submission template: `project/volume/data/raw/sample_submission.csv`
- Engineered features: `project/volume/data/interim/train_features.csv`, `project/volume/data/interim/test_features.csv`

## Response / Target

- `PM2.5_24h_ahead`

## Important Identifiers

- `observation_id` — row / submission ID
- `forecast_origin` — time of the forecast origin (used for lag joins and temporal holdout; not a predictor)
- `target_datetime` — hour being predicted (not a predictor)
- `station` — monitoring site (encoded as one-hot features; raw string not used as a numeric predictor)
- `row_set` — train/test marker created during feature building (dropped before modeling)

## Evaluation

- Root mean squared error (RMSE)
- Tuning uses a temporal holdout: fit on `forecast_origin < 2015-03-01`, validate on later hours
- Null baseline: always predict the mean `PM2.5_24h_ahead` from the fit period

## Required Outputs

- Prediction file: `project/volume/data/processed/submission.csv`
  - Required columns: `observation_id`, `PM2.5_24h_ahead`
- Saved model: `project/volume/models/pm25_xgb.model`

## Project Constraints

- Do not use information unavailable at forecast time (never lag or otherwise leak `PM2.5_24h_ahead` into features).
- Raw data under `project/volume/data/raw/` must not be modified in place.
- Scripts use repository-root-relative paths beginning with `project/`.
- Final workflow is declared in `project/entry_point.sh`:
  1. `project/src/features/build_features.R`
  2. `project/src/models/train_model.R`
- Holdout must be time-based (not random rows) to avoid leakage across hours.
- Predictions are clipped at 0 (PM2.5 cannot be negative).
- Random seed is fixed at `380`.

## Submission / Delivery Requirements

- Write `submission.csv` in `sample_submission` `observation_id` order.
- Submit the prediction file to Kaggle.
- Keep `entry_point.sh` aligned with the reproducible final workflow.

## Additional Notes

- Features include citywide same-hour means, 24h/48h lags, `PM10` change and station-vs-city gaps, cyclical hour/month encodings, weekday, and one-hot station / wind direction (with a missing-wind indicator).
- Final model is XGBoost (`reg:squarederror`) with early stopping on the temporal holdout, then refit on all training rows.
