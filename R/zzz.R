nf <- NULL

.onLoad <- function(libname, pkgname) {

  reticulate::use_virtualenv("nixtla-dev")

  nf <<- reticulate::import("neuralforecast", delay_load = TRUE)

  make_kan()
}
