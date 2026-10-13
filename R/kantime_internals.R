# initial functions from nixtla-r-tutorials ----

# model specs --
kan_spec <- \(h = 12L, input_size = 24L, max_steps=20L, freq = "ME", futr_exog_list = NULL){
  # reticulate: ensure character vector becomes a Python list of strings,
  # not a single string (which Python would iterate as characters).
  # Empty (character(0)) means "no xregs" -> keep NULL for python.
  if (!is.null(futr_exog_list) && length(futr_exog_list) > 0L) {
    futr_exog_list <- as.list(futr_exog_list)
  } else {
    futr_exog_list <- NULL
  }
  # UI: coerce numerics to integers for python (R users write 12 not 12L)
  h <- as.integer(h)
  input_size <- as.integer(input_size)
  max_steps <- as.integer(max_steps)
  nf$NeuralForecast(
    models = list(
      nf$models$KAN(
          h=h,
          input_size=input_size,
          loss=nf$losses$pytorch$RMSE(),
          max_steps=max_steps,
          scaler_type='standard',
          futr_exog_list=futr_exog_list
      )
    ),
    freq=freq
  )

}

# suppress noisy python warnings during fit/predict (e.g. pytorch lightning
# val_check_steps, nixtla id-column notices). issue #13.
# Note: lightning resets filters on Trainer init, so LeafSpec DeprecationWarning
# may still leak through — harmless, not the crash cause.
suppress_py_warnings <- \() {
  reticulate::py_run_string(paste(
    "import warnings",
    "warnings.filterwarnings('ignore', message='.*val_check_steps.*')",
    "warnings.filterwarnings('ignore', message='.*NIXTLA_ID_AS_COL.*')",
    "warnings.filterwarnings('ignore', message='.*LeafSpec.*')",
    "warnings.filterwarnings('ignore', message='.*TreeSpec.*')",
    "warnings.filterwarnings('ignore', module='.*_pytree.*')",
    "warnings.filterwarnings('ignore', category=DeprecationWarning)",
    "warnings.filterwarnings('ignore', category=FutureWarning)",
    sep = "\n"
  ))
  invisible(NULL)
}

# Extract a point-forecast column from neuralforecast predict() output.
# reticulate auto-converts pandas DataFrames to R data.frames, so handle
# both: R data.frame (preds[[model]]) and python objects (preds$MODEL$tolist()).
extract_forecast_col <- \(preds_forecast, model = c("KAN", "RMoK")) {
  model <- match.arg(model)
  if (inherits(preds_forecast, "data.frame")) {
    if (!model %in% names(preds_forecast)) {
      cli::cli_abort("Forecast output has no column {.code {model}} (got: {paste(names(preds_forecast), collapse=', ')}).")
    }
    return(as.numeric(preds_forecast[[model]]))
  }
  # python pandas object path
  as.numeric(reticulate::py_to_r(preds_forecast[[model]]$tolist()))
}

# Extract in-sample fitted values from predict_insample() output.
# Returns a numeric vector aligned to training rows, or NA vector if
# alignment is not possible (predict_insample is rolling, longer than train).
extract_insample_col <- \(insample, n) {
  if (inherits(insample, "data.frame")) {
    cols <- names(insample)
    model_col <- cols[!cols %in% c("unique_id", "ds", "y", "cutoff")][1]
    if (is.na(model_col)) {
      return(rep(NA_real_, n))
    }
    vec <- suppressWarnings(as.numeric(insample[[model_col]]))
    # rolling insample is longer than training; only use if lengths match,
    # otherwise fall back to NA (modeltime requires length n or 1)
    if (length(vec) != n) {
      return(rep(NA_real_, n))
    }
    return(vec)
  }
  # python pandas object path
  py_cols <- reticulate::py_to_r(insample$columns$tolist())
  model_col <- py_cols[!py_cols %in% c("unique_id", "ds", "y", "cutoff")][1]
  vec <- reticulate::py_to_r(insample[model_col])
  vec <- suppressWarnings(as.numeric(vec))
  if (length(vec) != n) {
    return(rep(NA_real_, n))
  }
  vec
}

# fit model with given specs ---
conformal_fit <- \(model_spec = NULL, df = NULL, outcome = "y", conformal_method = "conformal_error"){
  suppress_py_warnings()

  # like `outcome`,
  # should do the same for ds and unique_id,
  # in fact, i think unique_id should be set in here and masked,
  # these are local, not global models,
  # any map/lapply/loop should be on the modeltime side
  # i.e. a nested or group_by workflow

df <- df |> dplyr::rename(y = !!rlang::sym(outcome))

  # reticulate converts R Date to pandas object dtype; neuralforecast needs datetime
  py <- reticulate::py
  py$kantime_df <- reticulate::r_to_py(df)
  reticulate::py_run_string(
    "import pandas as pd\nkantime_df['ds'] = pd.to_datetime(kantime_df['ds'])"
  )
  py_df <- py$kantime_df

  model_spec$fit(df = py_df,
                 # by default make kan have conformal preds
                 prediction_intervals = nf$utils$PredictionIntervals(method=conformal_method))
}

# prediction function ---
conformal_predict <- \(model_spec = NULL, model_fit = NULL, level = NULL, futr_df = NULL){
  suppress_py_warnings()
  if (is.null(futr_df)) {
    model_spec$predict(model_fit, level = list(level))
  } else {
    # future exogenous values required when the model was trained with xregs
    # (fix ds dtype: reticulate converts R Date to object)
    py <- reticulate::py
    py$kantime_futr <- reticulate::r_to_py(futr_df)
    reticulate::py_run_string(
      "import pandas as pd\nkantime_futr['ds'] = pd.to_datetime(kantime_futr['ds'])"
    )
    py_futr <- py$kantime_futr
    model_spec$predict(model_fit, futr_df = py_futr, level = list(level))
  }
}

# Prepare a data.frame for neuralforecast: ensure unique_id, ds, y.
# Any other columns are treated as exogenous regressors by neuralforecast.
# Returns list(df, xregs) where xregs are the exogenous column names.
prepare_nixtla_df <- \(df, outcome = "y") {
  df <- df |> dplyr::rename(y = !!rlang::sym(outcome))

  # date column -> ds (first Date/POSIXct column)
  date_cols <- names(df)[vapply(df, function(col) inherits(col, c("Date", "POSIXct", "POSIXlt")), logical(1))]
  if (length(date_cols) == 0L) {
    cli::cli_abort("Could not find a date column for {.code ds}.")
  }
  ds_col <- date_cols[1]
  if (ds_col != "ds") {
    df <- df |> dplyr::rename(ds = !!rlang::sym(ds_col))
  }

  # unique_id: create a single dummy series if missing (local models)
  if (!"unique_id" %in% names(df)) {
    df$unique_id <- "series_1"
  }

  xregs <- setdiff(names(df), c("unique_id", "ds", "y"))
  list(df = df, xregs = xregs)
}

# Build the future dataframe for predict() from new_data.
# new_data has the date column and xreg values for the forecast horizon.
prepare_futr_df <- \(new_data, xregs) {
  date_cols <- names(new_data)[vapply(new_data, function(col) inherits(col, c("Date", "POSIXct", "POSIXlt")), logical(1))]
  if (length(date_cols) == 0L) {
    cli::cli_abort("Could not find a date column for {.code ds} in {.code new_data}.")
  }
  ds_col <- date_cols[1]
  futr_df <- new_data
  if (ds_col != "ds") {
    futr_df <- futr_df |> dplyr::rename(ds = !!rlang::sym(ds_col))
  }
  if (!"unique_id" %in% names(futr_df)) {
    futr_df$unique_id <- "series_1"
  }
  # keep only what neuralforecast needs, in a stable order
  futr_df[, c("unique_id", "ds", xregs), drop = FALSE]
}

# RMoK model specs --
rmok_spec <- \(h = 12L, input_size = 24L, max_steps=20L, freq = "ME"){
  # UI: coerce numerics to integers for python (R users write 12 not 12L)
  h <- as.integer(h)
  input_size <- as.integer(input_size)
  max_steps <- as.integer(max_steps)
  nf$NeuralForecast(
    models = list(
      nf$models$RMoK(
          h=h,
          input_size=input_size,
          n_series=1L,
          loss=nf$losses$pytorch$RMSE(),
          max_steps=max_steps,
          scaler_type='standard'
      )
    ),
    freq=freq
  )

}
