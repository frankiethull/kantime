#' Create Virtual Environment Wrapper
#'
#' @param envname virtual environment to create
#' @param ... additional passes for `create_virtualenv`
#'
#' @return creation of virtual environment
#' @export
create_neuralforecast_env <- \(envname = "nixtla-dev", ...){
  reticulate::create_virtualenv(envname, ...)
}
