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

   nixtla_df <- cbind(outcome, predictors)

   # neuralforecast format: unique_id, ds, y; other columns become xregs
   prepared  <- kantime:::prepare_nixtla_df(nixtla_df, outcome = "outcome")
   nixtla_df <- prepared$df
   xreg_cols <- prepared$xregs

   model_spec <- kantime:::kan_spec(h = h,
                                    input_size = input_size,
                                    max_steps  = max_steps,
                                    freq       = freq,
                                    futr_exog_list = xreg_cols)

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
    extras = list(xregs = xreg_cols),
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

  # XREG: future exogenous values for the forecast horizon
  xreg_cols <- object$extras$xregs
  futr_df <- NULL
  if (length(xreg_cols) > 0L) {
    futr_df <- kantime:::prepare_futr_df(new_data, xreg_cols)
  }

  # PREDICTIONS, CONFORMAL PREDS
  preds_forecast <- kantime:::conformal_predict(
                       model_spec = specs,
                       model_fit = model,
                       level = level,
                       futr_df = futr_df)

  # Return predictions as numeric vector
  preds <- as.numeric(preds_forecast$KAN$tolist())
  # nixtla's conformals, include for type = "prob"
  # pred_lower <- preds_forecast[,3]
  # pred_upper <- preds_forecast[,4]
  return(preds)

}


# ---
# RMOK (Reversible Mixture of KAN) ----
# ---

#' @export
rmok <- \(
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
    "rmok",
    args     = args,
    eng_args = NULL,
    mode     = mode,
    method   = NULL,
    engine   = NULL
  )

}


# Define the fitting bridge function
#' @export
rmok_bridge_fit_impl <- \(x, y, h, input_size, max_steps, freq, scale = NULL) {

   outcome    <- y # Comes in as a vector
   predictors <- x # Comes in as a data.frame (dates and possible xregs)

   nixtla_df <- cbind(outcome, predictors)

   # neuralforecast format: unique_id, ds, y; other columns become xregs
   prepared  <- kantime:::prepare_nixtla_df(nixtla_df, outcome = "outcome")
   nixtla_df <- prepared$df
   xreg_cols <- prepared$xregs

   # RMoK does not support exogenous regressors in neuralforecast 3.3.0
   # (EXOGENOUS_FUTR/HIST/STAT are all False); fail loudly instead of
   # silently ignoring them.
   if (length(xreg_cols) > 0L) {
     cli::cli_abort("RMoK does not support exogenous regressors {.code ({xreg_cols})}. Use KAN instead.")
   }

   model_spec <- kantime:::rmok_spec(h = h,
                                     input_size = input_size,
                                     max_steps  = max_steps,
                                     freq       = freq)

    model_fit  <-  kantime:::conformal_fit(
                     model_spec = model_spec,
                     df = nixtla_df,
                     outcome = "y",
                     conformal_method = "conformal_error"
                  )

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
    class = "rmok_bridge_fit_impl",
    models = list(model_fit = model_fit, model_spec = model_spec),
    data = modeltime_table,
    extras = list(xregs = xreg_cols),
    desc = "RMoK Model"
  )
}


# Define the bridge's predict method
#' @export
predict.rmok_bridge_fit_impl <- \(object, new_data, level = 90L, ...) {

  # PREPARE INPUTS
  model       <- object$models$model_fit

  # SPECS
  specs       <- object$models$model_spec

  # XREG: future exogenous values for the forecast horizon
  xreg_cols <- object$extras$xregs
  futr_df <- NULL
  if (length(xreg_cols) > 0L) {
    futr_df <- kantime:::prepare_futr_df(new_data, xreg_cols)
  }

  # PREDICTIONS, CONFORMAL PREDS
  preds_forecast <- kantime:::conformal_predict(
                       model_spec = specs,
                       model_fit = model,
                       level = level,
                       futr_df = futr_df)

  # Return predictions as numeric vector
  preds <- as.numeric(preds_forecast$RMoK$tolist())
  return(preds)

}
