nf <- NULL

.onLoad <- function(libname, pkgname) {
  reticulate::py_require("neuralforecast")

  nf <<- reticulate::import("neuralforecast", delay_load = TRUE)

  make_kan()
  make_rmok()
}
