# Nixtla's KAN is based on modeltime.gluonts thx to Matt Dancho :P
make_kan <- \(){

  # SETUP
  model <- "kan"
  mode  <- "regression"
  eng   <- "kan"

  parsnip::set_new_model(model)
  parsnip::set_model_mode(model, mode)

  # KAN: regression ----

  # * Model ----
  parsnip::set_model_engine(model, mode = mode, eng = eng)
  parsnip::set_dependency(model, eng = eng, pkg = "reticulate")
  parsnip::set_dependency(model, eng = eng, pkg = "kantime")



  # * KAN Args ----
  parsnip::set_model_arg(
    model        = model,
    eng          = eng,
    parsnip      = "h",
    original     = "h",
    func         = list(pkg = "kantime", fun = "h"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model        = model,
    eng          = eng,
    parsnip      = "input_size",
    original     = "input_size",
    func         = list(pkg = "kantime", fun = "input_size"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model        = model,
    eng          = eng,
    parsnip      = "max_steps",
    original     = "max_steps",
    func         = list(pkg = "kantime", fun = "max_steps"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model        = model,
    eng          = eng,
    parsnip      = "freq",
    original     = "freq",
    func         = list(pkg = "kantime", fun = "freq"),
    has_submodel = FALSE
  )

  # * Encoding ----
  parsnip::set_encoding(
    model   = model,
    eng     = eng,
    mode    = mode,
    options = list(
      predictor_indicators = "none",
      compute_intercept    = FALSE,
      remove_intercept     = FALSE,
      allow_sparse_x       = FALSE
    )
  )


  # * Fit ----
  parsnip::set_fit(
    model         = model,
    eng           = eng,
    mode          = mode,
    value         = list(
      interface = "data.frame",
      protect   = c("x", "y"),
      func      = c(pkg = "kantime", fun = "kan_bridge_fit_impl"),
      defaults  = list()
    )
  )

  # * Predict ----
  parsnip::set_pred(
    model         = model,
    eng           = eng,
    mode          = mode,
    type          = "numeric",
    value         = list(
      pre       = NULL,
      post      = NULL,
      func      = c(fun = "predict"),
      args      =
        list(
          object   = rlang::expr(object$fit),
          new_data = rlang::expr(new_data)
        )
    )
  )
}
