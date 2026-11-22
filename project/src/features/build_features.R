library(data.table)  # method: data.table

# method: fread. Paths are relative to the Beijing PM2.5 repo root.
train <- fread("project/volume/data/raw/STAT_380_Train_P3.csv")
test <- fread("project/volume/data/raw/STAT_380_Test_P3.csv")

# method: label column, not a feature. Stack so lags and city means share one clock.
# row_set is NOT a predictor (dropped later in train_model.R).
train[, row_set := "train"]
test[, row_set := "test"]
master <- rbindlist(list(train, test), use.names = TRUE, fill = TRUE)  # method: row-bind
setorder(master, station, forecast_origin)  # method: sort by station, then time

# ---------------------------------------------------------------------------
# method: group mean. Feature: citywide average at this same hour
# ---------------------------------------------------------------------------
# New columns: city_PM10, city_SO2, city_NO2, city_CO, city_O3,
#              city_TEMP, city_PRES, city_DEWP, city_RAIN, city_WSPM
# Built from: the matching raw column (PM10 -> city_PM10, etc.)
# Why OK at forecast time: all 12 stations are observed together.
# NOT built from the target PM2.5_24h_ahead.
num_cols <- c("PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM")
city_cols <- paste0("city_", num_cols)
master[, (city_cols) := lapply(.SD, function(x) mean(x, na.rm = TRUE)),  # method: mean by group
       by = forecast_origin, .SDcols = num_cols]

# ---------------------------------------------------------------------------
# method: lag. Features: same station, 24h and 48h earlier
# ---------------------------------------------------------------------------
# New columns: <source>_lag24 and <source>_lag48 for each name in lag_src.
# Built from: PM10, NO2, CO, O3, TEMP, DEWP, WSPM, city_PM10, city_CO, city_NO2
# How: shift forecast_origin forward by h*3600 seconds, then merge back.
#   A row at time t picks up the values that lived at time t - h.
# Never lagged: PM2.5_24h_ahead (that is Y, not a feature).
lag_src <- c("PM10", "NO2", "CO", "O3", "TEMP", "DEWP", "WSPM", "city_PM10", "city_CO", "city_NO2")
for (h in c(24L, 48L)) {
  lagged <- master[, c("station", "forecast_origin", lag_src), with = FALSE]
  lagged[, forecast_origin := forecast_origin + h * 3600]  # method: time shift (lag)
  setnames(lagged, lag_src, paste0(lag_src, "_lag", h))    # method: rename to *_lag24 / *_lag48
  master <- merge(master, lagged, by = c("station", "forecast_origin"), all.x = TRUE)  # method: left join
}

# ---------------------------------------------------------------------------
# method: difference. Features: 24h change and station-vs-city gap
# ---------------------------------------------------------------------------
master[, PM10_change_24 := PM10 - PM10_lag24]  # feature from: PM10, PM10_lag24
master[, PM10_vs_city := PM10 - city_PM10]     # feature from: PM10, city_PM10

# method: cyclical encoding (sin/cos). Hour 23 neighbors hour 0; December neighbors January.
# hour_sin, hour_cos from: hour. month_sin, month_cos from: month.
master[, hour_sin := sin(2 * pi * hour / 24)]
master[, hour_cos := cos(2 * pi * hour / 24)]
master[, month_sin := sin(2 * pi * month / 12)]
master[, month_cos := cos(2 * pi * month / 12)]
master[, wday := as.POSIXlt(forecast_origin)$wday]  # method: weekday from timestamp. feature: wday

# ---------------------------------------------------------------------------
# method: one-hot / dummy encoding. xgboost needs numeric X, not factors.
# ---------------------------------------------------------------------------
stations <- sort(unique(master$station))
winds <- sort(unique(master$wd[!is.na(master$wd)]))
for (s in stations) {
  master[, paste0("st_", s) := as.integer(station == s)]  # feature: st_<station> from station
}
master[, wd_missing := as.integer(is.na(wd))]  # method: missing indicator. feature: wd_missing from wd
for (w in winds) {
  master[, paste0("wd_", w) := as.integer(!is.na(wd) & wd == w)]  # feature: wd_<direction> from wd
}

# ---------------------------------------------------------------------------
# Which columns become X, and which column is Y
# ---------------------------------------------------------------------------
# method: keep numeric columns, then drop ids and the response.
# DROP (not predictors):
#   year, month, day, hour     raw clock (replaced by sin/cos and wday)
#   PM2.5_24h_ahead            this is Y
#   forecast_origin            time id (lag join and holdout cut)
#   target_datetime            id of the hour being predicted
# pred_cols is the X list written out for train_model.R.
# Left in X: raw numerics (PM10, gases, weather, ...) PLUS every feature above.
num_flags <- vapply(master, is.numeric, logical(1))
drop_cols <- c(
  "year", "month", "day", "hour", "PM2.5_24h_ahead",
  "forecast_origin", "target_datetime"
)
num_flags[drop_cols] <- FALSE
pred_cols <- names(master)[num_flags]
for (v in pred_cols) {
  master[!is.finite(get(v)), (v) := NA_real_]  # method: turn NaN/Inf into NA (missing)
}

# Y stays on the training file only. Test has no PM2.5_24h_ahead.
keep <- c("observation_id", "forecast_origin", "row_set", "PM2.5_24h_ahead", pred_cols)
train_out <- master[row_set == "train", ..keep]  # method: filter to train rows
test_out <- master[row_set == "test", ..keep]
test_out[, PM2.5_24h_ahead := NULL]

fwrite(train_out, "project/volume/data/interim/train_features.csv")  # method: fwrite
fwrite(test_out, "project/volume/data/interim/test_features.csv")
cat("Feature columns:", length(pred_cols), "\n")
