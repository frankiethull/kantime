# initial functions from nixtla-r-tutorials ----

# model specs --
kan_spec <- \(h = 12L, input_size = 24L, max_steps=20L, freq = "ME", futr_exog_list = NULL){
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

# fit model with given specs ---
conformal_fit <- \(model_spec = NULL, df = NULL, outcome = "y", conformal_method = "conformal_error"){

  # like `outcome`,
  # should do the same for ds and unique_id,
  # in fact, i think unique_id should be set in here and masked,
  # these are local, not global models,
  # any map/lapply/loop should be on the modeltime side
  # i.e. a nested or group_by workflow

df <- df |> dplyr::rename(y = !!rlang::sym(outcome))

  model_spec$fit(df = df,
                 # by default make kan have conformal preds
                 prediction_intervals = nf$utils$PredictionIntervals(method=conformal_method))
}

# prediction function ---
conformal_predict <- \(model_spec = NULL, model_fit = NULL, level = NULL, futr_df = NULL){
  if (!is.null(futr_df)) {
    # Future exogenous regressors provided
    model_spec$predict(model_fit, futr_df = futr_df, level = list(level))
  } else {
    model_spec$predict(model_fit, level = list(level))
  }
}
