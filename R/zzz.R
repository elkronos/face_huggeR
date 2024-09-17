#' Package Initialization
#'
#' This function is called when the package is loaded. It performs
#' necessary initialization tasks, such as setting up the Python
#' environment.
#'
#' @param libname A character string representing the library path.
#' @param pkgname A character string representing the name of the package.
#'
#' @details
#' When the package is loaded, a message indicating that the package
#' is being loaded is printed, and the \code{setup_python_environment()}
#' function is called automatically to ensure the Python environment
#' is ready for use.
#'
#' @export
.onLoad <- function(libname, pkgname) {
  message("Loading MyPackage...")
  setup_python_environment()
}
