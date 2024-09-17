#' Setup Python Environment using Miniconda and Reticulate
#'
#' This script provides a set of functions to automate the setup of a Python environment
#' using Miniconda and the `reticulate` package in R. It includes functionalities to
#' check and install Miniconda, create and manage Conda environments, install Python
#' packages, and import Python modules into R.
#'
#' @section Functions:
#' \itemize{
#'   \item \code{\link{get_miniconda_path}}
#'   \item \code{\link{is_miniconda_installed}}
#'   \item \code{\link{install_miniconda_if_missing}}
#'   \item \code{\link{conda_env_exists}}
#'   \item \code{\link{create_conda_env}}
#'   \item \code{\link{install_python_packages}}
#'   \item \code{\link{activate_conda_env}}
#'   \item \code{\link{verify_python_config}}
#'   \item \code{\link{import_python_modules}}
#'   \item \code{\link{setup_python_environment}}
#' }
#'
#' @examples
#' \dontrun{
#' # To set up the default Python environment:
#' setup_python_environment()
#' }
#' @name python_env_setup
NULL

#' Get the Default Miniconda Installation Path
#'
#' Retrieves the default installation path of Miniconda managed by the `reticulate` package.
#'
#' @return A character string specifying the path to the Miniconda installation.
#' @export
#'
#' @examples
#' get_miniconda_path()
get_miniconda_path <- function() {
  reticulate::miniconda_path()
}

#' Check if Miniconda is Installed
#'
#' Determines whether Miniconda is installed by verifying the existence of the installation directory.
#'
#' @param miniconda_path A character string specifying the path to the Miniconda installation.
#'
#' @return A logical value: \code{TRUE} if Miniconda is installed, \code{FALSE} otherwise.
#' @export
#'
#' @examples
#' path <- get_miniconda_path()
#' is_miniconda_installed(path)
is_miniconda_installed <- function(miniconda_path) {
  dir.exists(miniconda_path)
}

#' Install Miniconda if Missing
#'
#' Checks if Miniconda is installed and installs it using `reticulate::install_miniconda()` if not present.
#'
#' @return None. Prints messages indicating the installation status.
#' @export
#'
#' @examples
#' install_miniconda_if_missing()
install_miniconda_if_missing <- function() {
  miniconda_path <- get_miniconda_path()
  
  if (!is_miniconda_installed(miniconda_path)) {
    message("Miniconda not found. Installing Miniconda...")
    reticulate::install_miniconda()
    message("Miniconda installed at: ", miniconda_path)
  } else {
    message("Miniconda is already installed at: ", miniconda_path)
  }
}

#' Check if a Conda Environment Exists
#'
#' Determines whether a specified Conda environment exists.
#'
#' @param env_name A character string specifying the name of the Conda environment to check.
#'
#' @return A logical value: \code{TRUE} if the Conda environment exists, \code{FALSE} otherwise.
#' @export
#'
#' @examples
#' conda_env_exists("r-reticulate")
conda_env_exists <- function(env_name) {
  conda_envs <- tryCatch(
    reticulate::conda_list(),
    error = function(e) {
      warning("Failed to retrieve Conda environments: ", e$message)
      return(data.frame())  # Return empty data frame on error
    }
  )
  
  return(env_name %in% conda_envs$name)
}

#' Create a Conda Environment
#'
#' Creates a new Conda environment with the specified name and Python version if it does not already exist.
#'
#' @param env_name A character string specifying the name of the Conda environment to create.
#' @param python_version A character string specifying the Python version to install (default is "3.8").
#'
#' @return None. Prints messages indicating the creation status of the Conda environment.
#' @export
#'
#' @examples
#' create_conda_env("my-env", python_version = "3.9")
create_conda_env <- function(env_name, python_version = "3.8") {
  if (!conda_env_exists(env_name)) {
    message("Creating Conda environment '", env_name, "' with Python ", python_version, "...")
    reticulate::conda_create(envname = env_name, packages = paste0("python=", python_version))
    message("Conda environment '", env_name, "' created successfully.")
  } else {
    message("Conda environment '", env_name, "' already exists.")
  }
}

#' Install Python Packages into a Conda Environment
#'
#' Installs specified Python packages into a given Conda environment using `conda_install`.
#'
#' @param env_name A character string specifying the name of the Conda environment.
#' @param packages A character vector of Python package names to install.
#' @param use_pip A logical value indicating whether to use `pip` for installation (default is \code{TRUE}).
#'
#' @return None. Prints messages indicating the installation status of Python packages.
#' @export
#'
#' @examples
#' install_python_packages("my-env", c("numpy", "pandas"), use_pip = TRUE)
install_python_packages <- function(env_name, packages, use_pip = TRUE) {
  message("Installing Python packages into Conda environment '", env_name, "'...")
  reticulate::conda_install(envname = env_name, packages = packages, pip = use_pip)
  message("Python packages installed successfully in '", env_name, "'.")
}

#' Activate a Conda Environment in Reticulate
#'
#' Activates the specified Conda environment within the current R session using `reticulate::use_condaenv`.
#'
#' @param env_name A character string specifying the name of the Conda environment to activate.
#'
#' @return None. Prints a message confirming the activation of the Conda environment.
#' @export
#'
#' @examples
#' activate_conda_env("my-env")
activate_conda_env <- function(env_name) {
  reticulate::use_condaenv(env_name, required = TRUE)
  message("Activated Conda environment '", env_name, "'.")
}

#' Verify Python Configuration
#'
#' Retrieves and prints the current Python configuration used by `reticulate`.
#'
#' @return None. Prints the Python configuration details.
#' @export
#'
#' @examples
#' verify_python_config()
verify_python_config <- function() {
  py_config <- reticulate::py_config()
  message("Python configuration:")
  print(py_config)
}

#' Import Python Modules and Confirm Their Loading
#'
#' Imports specified Python modules into the R session using `reticulate::import` and verifies their successful loading.
#'
#' @param modules A character vector of Python module names to import.
#'
#' @return A named list of imported Python modules. Each element corresponds to a module; \code{NULL} if the import failed.
#' @export
#'
#' @examples
#' import_python_modules(c("numpy", "pandas"))
import_python_modules <- function(modules) {
  imported_modules <- list()
  
  for (module in modules) {
    imported_modules[[module]] <- tryCatch(
      reticulate::import(module, delay_load = TRUE),
      error = function(e) {
        warning("Failed to import module '", module, "': ", e$message)
        return(NULL)
      }
    )
  }
  
  # Check if all modules are successfully imported
  if (all(sapply(imported_modules, Negate(is.null)))) {
    message("Successfully imported all specified Python modules: ", paste(modules, collapse = ", "), ".")
  } else {
    stop("Failed to import one or more Python modules. Please check the installation.")
  }
  
  return(imported_modules)
}

#' Master Function to Set Up the Python Environment
#'
#' Orchestrates the entire process of setting up a Python environment by installing Miniconda,
#' creating a Conda environment, installing Python packages, activating the environment, verifying
#' the Python configuration, and importing specified Python modules.
#'
#' @param env_name A character string specifying the name of the Conda environment to create or use (default is "r-reticulate").
#' @param python_version A character string specifying the Python version to install (default is "3.8").
#' @param python_packages A character vector of Python package names to install via `pip` (default is \code{c("transformers", "torch")}).
#' @param modules_to_import A character vector of Python module names to import into R (default is \code{c("transformers", "torch")}).
#'
#' @return None. Prints messages indicating the progress and completion of the Python environment setup.
#' @export
#'
#' @examples
#' setup_python_environment()
#' 
#' # Custom setup example:
#' setup_python_environment(
#'   env_name = "my-custom-env",
#'   python_version = "3.9",
#'   python_packages = c("numpy", "pandas", "scikit-learn"),
#'   modules_to_import = c("numpy", "pandas", "sklearn")
#' )
setup_python_environment <- function(
    env_name = "r-reticulate",
    python_version = "3.8",
    python_packages = c("transformers", "torch"),
    modules_to_import = c("transformers", "torch")
) {
  install_miniconda_if_missing()
  create_conda_env(env_name, python_version)
  install_python_packages(env_name, python_packages, use_pip = TRUE)
  activate_conda_env(env_name)
  verify_python_config()
  import_python_modules(modules_to_import)
  
  message("Python environment setup is complete.")
}

# Example usage:
# Uncomment the line below to run the setup
# setup_python_environment()