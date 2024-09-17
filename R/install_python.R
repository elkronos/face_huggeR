# Install packages
required_packages <- c("reticulate", "progress", "httr", "tibble", "tm", "data.table")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# Get the default Miniconda installation path
miniconda_path <- reticulate::miniconda_path()

# Function to check if Miniconda is installed
is_miniconda_installed <- function(miniconda_path) {
  dir.exists(miniconda_path)
}

# Install Miniconda if not installed
if (!is_miniconda_installed(miniconda_path)) {
  cat("Installing Miniconda...\n")
  reticulate::install_miniconda()
} else {
  cat("Miniconda is already installed at:", miniconda_path, "\n")
}

# Create Conda Environment and Install Python Packages
env_name <- "r-reticulate"
python_packages <- c("transformers", "torch")

# Check if the Conda environment exists
conda_envs <- tryCatch({
  reticulate::conda_list()
}, error = function(e) {
  data.frame()  # Return an empty data frame if conda_list() fails
})

if (!(env_name %in% conda_envs$name)) {
  cat("Creating Conda environment '", env_name, "'...\n", sep = "")
  reticulate::conda_create(envname = env_name, packages = "python=3.8")
} else {
  cat("Conda environment '", env_name, "' already exists.\n", sep = "")
}

# Install required Python packages into the Conda environment
cat("Installing Python packages into Conda environment '", env_name, "'...\n", sep = "")
reticulate::conda_install(envname = env_name, packages = python_packages, pip = TRUE)

# Use the Conda Environment in reticulate
reticulate::use_condaenv(env_name, required = TRUE)

# Verify Python configuration
py_config <- reticulate::py_config()
cat("Python configuration:\n")
print(py_config)

# Import Python modules and confirm they are loaded
transformers <- reticulate::import("transformers", delay_load = TRUE)
torch <- reticulate::import("torch", delay_load = TRUE)

if (!is.null(transformers) && !is.null(torch)) {
  cat("Successfully imported 'transformers' and 'torch'.\n")
} else {
  stop("Failed to import 'transformers' or 'torch'.")
}
