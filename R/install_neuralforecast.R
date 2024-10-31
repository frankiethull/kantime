#' Install Development Version of neuralforecast From Github dev version contains conformal methods for predicting
#'
#' @param envname virtual environment name
#' @param method  method defaults to "auto"
#' @param ...     additional passes for `py_install`
#'
#' @return worth the wait! installs neuralforecast to "nixtla-dev" by default
#' @export
install_neuralforecast <- \(envname = "nixtla-dev", method = "auto", ...) {
  # reticulate::py_install("neuralforecast", envname = envname,
  #                        method = method, pip = TRUE, ...)
  #
  # conformal predictions ~
  reticulate::py_install("git+https://github.com/Nixtla/neuralforecast.git", envname = envname,
                         method = method, ...)

}
