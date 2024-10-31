#' Use Virtual Environment Wrapper
#'
#' @param envname virtual environment to use
#' @param ...  additional passes for `use_virtualenv`
#'
#' @return sets environment to virtualenv
#' @export
use_neuralforecast_env <- \(envname = "nixtla-dev", ...){
reticulate::use_virtualenv(envname, ...)
}
