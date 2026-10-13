#' @export
kan <- \(
    mode = "regression",

    # Required Args
    h,
    input_size,
    max_steps,
    freq,

    # Modeltime Args
    scale = NULL


) {

  args <- list(
    # Required Args
    h                      = rlang::enquo(h),
    input_size             = rlang::enquo(input_size),
    max_steps              = rlang::enquo(max_steps),
    freq                   = rlang::enquo(freq),

    # Modeltime Args
    scale                  = rlang::enquo(scale)
  )

  parsnip::new_model_spec(
    "kan",
    args     = args,
    eng_args = NULL,
    mode     = mode,
    method   = NULL,
    engine   = NULL
  )

}


# Define the fitting bridge function
#' @export
kan_bridge_fit_impl <- \(x, y, h, input_size, max_steps, freq, scale = NULL) {

   outcome    <- y # Comes in as a vector
   predictors <- x # Comes in as a data.frame (dates and possible xregs)

   # Identify date column and xreg columns
   # The date column is the one that parses as a date/time
   date_col <- NULL
   xreg_cols <- character(0)
   for (col in names(predictors)) {
     if (inherits(predictors[[col]], c("Date", "POSIXct", "POSIXlt"))) {
       date_col <- col
     } else if (is.numeric(predictors[[col]])) {
       xreg_cols <- c(xreg_cols, col)
     }
   }
   # Fallback: if no date column found, use the first column
   if (is.null(date_col)) {
     date_col <- names(predictors)[1]
   }

   # Build nixtla dataframe with unique_id, ds, y, and xregs
   # NB: convert Date to POSIXct for proper reticulate -> pandas conversion
   ds_vals <- predictors[[date_col]]
   if (inherits(ds_vals, "Date")) {
     ds_vals <- as.POSIXct(ds_vals)
   }
   nixtla_df <- data.frame(
     unique_id = "series_1",
     ds = ds_vals,
     y = outcome,
     stringsAsFactors = FALSE
   )
   # Add xreg columns
   for (col in xreg_cols) {
     nixtla_df[[col]] <- predictors[[col]]
   }

   model_spec <- kantime:::kan_spec(h = h,
                                    input_size = input_size,
                                    max_steps  = max_steps,
                                    freq       = freq,
                                    futr_exog_list = as.list(xreg_cols))

    model_fit  <-  kantime:::conformal_fit(
                     model_spec = model_spec,
                     df = nixtla_df,
                     outcome = "y",
                     conformal_method = "conformal_error"
                  )

# predict_insample TODO:
    # fitted_values <- model_spec$predict_insample(step_size = 1L)
  fitted_values <- NA

    # Create the modeltime table
    modeltime_table <- tibble::tibble(
                        modeltime::parse_index_from_data(x)
                        ) |>
                        dplyr::mutate(
                        .actual = outcome,
                        .fitted = fitted_values,
                        .residuals = .actual - .fitted
                      )

  # Create the modeltime bridge
  modeltime::new_modeltime_bridge(
    class = "kan_bridge_fit_impl",
    models = list(model_fit = model_fit, model_spec = model_spec),
    data = modeltime_table,
    extras = list(
      xreg_cols = xreg_cols,
      date_col = date_col
    ),
    desc = "KAN Model"
  )
}


# Define the bridge's predict method
#' @export
predict.kan_bridge_fit_impl <- \(object, new_data, level = 90L, ...) {

  # PREPARE INPUTS
  model       <- object$models$model_fit

  # SPECS
  specs       <- object$models$model_spec

  # XREG: build future dataframe with exogenous regressors if available
  xreg_cols <- object$extras$xreg_cols
  date_col <- object$extras$date_col
  future_df <- NULL
  if (!is.null(xreg_cols) && length(xreg_cols) > 0 && !is.null(new_data)) {
    # new_data has future dates and xreg values
    # NB: convert Date to POSIXct for proper reticulate -> pandas conversion
    fut_ds <- new_data[[date_col]]
    if (inherits(fut_ds, "Date")) {
      fut_ds <- as.POSIXct(fut_ds)
    }
    future_df <- data.frame(
      unique_id = "series_1",
      ds = fut_ds,
      stringsAsFactors = FALSE
    )
    for (col in xreg_cols) {
      if (col %in% names(new_data)) {
        future_df[[col]] <- new_data[[col]]
      }
    }
  }

  # PREDICTIONS, CONFORMAL PREDS
  preds_forecast <- kantime:::conformal_predict(
                       model_spec = specs,
                       model_fit = model,
                       level = level,
                       futr_df = future_df)

  # Return predictions as numeric vector
  preds <- as.numeric(preds_forecast$KAN)
  # nixtla's conformals, include for type = "prob"
  # pred_lower <- preds_forecast[,3]
  # pred_upper <- preds_forecast[,4]
  return(preds)

}
