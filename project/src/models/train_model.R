library(data.table)  # method: data.table
library(xgboost)       # method: xgboost (gradient boosted trees). Not lm, not glmnet.

set.seed(380)  # method: fixed seed so the boosted trees are reproducible

# method: fread. Features come from build_features.R.
# sample_submission is only the id order for the output file.
train <- fread("project/volume/data/interim/train_features.csv")
test <- fread("project/volume/data/interim/test_features.csv")
submit <- fread("project/volume/data/raw/sample_submission.csv")

# ---------------------------------------------------------------------------
# X and Y for this script
# ---------------------------------------------------------------------------
# Y is PM2.5_24h_ahead (the label = argument below). No object is named y.
# X is pred_cols: every column except these ids. No object is named X.
#   observation_id, forecast_origin, row_set, PM2.5_24h_ahead
id_cols <- c("observation_id", "forecast_origin", "row_set", "PM2.5_24h_ahead")
pred_cols <- setdiff(names(train), id_cols)  # method: drop id columns to get X names

# method: temporal holdout. Not random k-fold, not profile CV.
# Fit on hours before 2015-03-01. Score hours on/after that date.
# Random rows would let later hours leak into the fit.
cut_time <- as.POSIXct("2015-03-01", tz = "UTC")
fit_rows <- train[forecast_origin < cut_time]   # method: filter. grows trees while tuning
val_rows <- train[forecast_origin >= cut_time]  # method: filter. held out; scored, not fit

to_matrix <- function(dt) {
  as.matrix(dt[, ..pred_cols])  # method: design matrix. columns = pred_cols, in this order
}

# method: xgb.DMatrix. label = Y. missing = NA marks missing feature cells.
dfit <- xgb.DMatrix(to_matrix(fit_rows), label = fit_rows[["PM2.5_24h_ahead"]], missing = NA)
dval <- xgb.DMatrix(to_matrix(val_rows), label = val_rows[["PM2.5_24h_ahead"]], missing = NA)

# method: xgboost hyperparameters. No lambda (L2) and no alpha (L1) are set.
# This is not a penalized regression. Knobs that ARE set:
#   objective reg:squarederror   loss = squared error (not MAE)
#   eval_metric rmse             score = root mean squared error
#   eta                          learning rate
#   max_depth                    tree depth
#   min_child_weight             minimum weight in a leaf before a split
#   subsample                    row sampling per tree
#   colsample_bytree             column sampling per tree
#   tree_method hist              histogram tree builder
#   nrounds                      number of trees, chosen below by early stopping
param <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.05,
  max_depth = 7,
  min_child_weight = 5,
  subsample = 0.8,
  colsample_bytree = 0.8,
  tree_method = "hist"
)

# method: xgb.train + early stopping. Fit on fit_rows only, watch val RMSE.
# Stop after 40 rounds with no val improvement. Not the model that is submitted.
set.seed(380)
tune_fit <- xgb.train(
  params = param,
  data = dfit,
  nrounds = 800,
  evals = list(fit = dfit, val = dval),
  early_stopping_rounds = 40,
  print_every_n = 50
)

best_nrounds <- attr(tune_fit, "early_stop")$best_iteration  # method: early stopping picks tree count
val_pred <- predict(tune_fit, dval)  # method: predict on the holdout
val_rmse <- sqrt(mean((val_rows[["PM2.5_24h_ahead"]] - val_pred)^2))  # method: RMSE
# method: null model. Always predict the mean Y from the fit period.
null_rmse <- sqrt(mean((val_rows[["PM2.5_24h_ahead"]] - mean(fit_rows[["PM2.5_24h_ahead"]]))^2))
cat("Trees:", best_nrounds, "\n")
cat("Holdout RMSE:", round(val_rmse, 2), "  null RMSE:", round(null_rmse, 2), "\n")

# ---------------------------------------------------------------------------
# method: refit. Same xgboost params, ALL training rows, nrounds = best_nrounds
# ---------------------------------------------------------------------------
# Final X = to_matrix(train)          columns = pred_cols
# Final Y = train[["PM2.5_24h_ahead"]]
# Test X  = to_matrix(test)           no label; test has no Y
dall <- xgb.DMatrix(to_matrix(train), label = train[["PM2.5_24h_ahead"]], missing = NA)
set.seed(380)
final_fit <- xgb.train(
  params = param,
  data = dall,
  nrounds = best_nrounds,
  verbose = 0
)

dtest <- xgb.DMatrix(to_matrix(test), missing = NA)
test[, PM2.5_24h_ahead := predict(final_fit, dtest)]  # method: predict. writes Y-hat
test[, PM2.5_24h_ahead := pmax(PM2.5_24h_ahead, 0)]   # method: clip at 0. PM2.5 cannot be negative

# method: left join. Keep sample_submission row order. Fill PM2.5_24h_ahead.
submit <- merge(
  submit[, .(observation_id)],
  test[, .(observation_id, PM2.5_24h_ahead)],
  by = "observation_id",
  all.x = TRUE,
  sort = FALSE
)

fwrite(submit, "project/volume/data/processed/submission.csv")  # method: fwrite
xgb.save(final_fit, "project/volume/models/pm25_xgb.model")     # method: save the boosted model
cat("Submission rows:", nrow(submit), " missing:", sum(is.na(submit[["PM2.5_24h_ahead"]])), "\n")
