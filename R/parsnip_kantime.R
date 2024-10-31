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

   model_spec <- kantime:::kan_spec(h = h,
                                    input_size = input_size,
                                    max_steps  = max_steps,
                                    freq       = freq)

    model_fit  <-  kantime:::conformal_fit(
                     model_spec = model_spec,
                     df = nixtla_df,
                     outcome = "outcome",
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
    extras = list(NULL), # add xregs TODO
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

  # XREG, currently not implemented
  # xreg_recipe <- object$extras$xreg_recipe
  # xreg_matrix <- bake_xreg_recipe(xreg_recipe, new_data, format = "matrix")

  # PREDICTIONS, CONFORMAL PREDS
  preds_forecast <- kantime:::conformal_predict(
                       model_spec = specs,
                       model_fit = model,
                       level = level)

  # Return predictions as numeric vector
  preds <- as.numeric(preds_forecast$KAN)
  # nixtla's conformals, include for type = "prob"
  # pred_lower <- preds_forecast[,3]
  # pred_upper <- preds_forecast[,4]
  return(preds)

}
