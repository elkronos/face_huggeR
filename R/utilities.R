required_packages <- c("tm", "stringr")

# Function to check if a package is installed, and install if missing
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}


#' Task Validator
#'
#' Validates a given task against a set of valid tasks. The function checks if the provided task
#' is part of a list of predefined tasks, and throws an error if the task is not supported.
#'
#' @param task Character string representing the task to be validated. Should match one of the valid tasks.
#' @param valid_tasks A character vector containing valid task names. If the `task` is not part of this list, the function will throw an error.
#'
#' @details
#' This function is useful in workflows where different tasks are supported (e.g., classification, generation).
#' It ensures that only valid tasks proceed in the workflow. If the task is not valid, an informative error message
#' is displayed.
#'
#' @examples
#' \dontrun{
#' valid_tasks <- c("classification", "generation")
#' validate_task("classification", valid_tasks)  # Passes
#' validate_task("invalid_task", valid_tasks)    # Throws an error
#' }
#' 
#' @export
validate_task <- function(task, valid_tasks) {
  cat("Validating task:", task, "\n")
  if (!(task %in% valid_tasks)) {
    stop(
      sprintf(
        "Unsupported task: '%s'. Supported tasks: %s.",
        task,
        paste(valid_tasks, collapse = ", ")
      )
    )
  }
  cat("Task validation passed:", task, "\n")
}

#' Device Validator
#'
#' Validates whether the provided device is either "cpu" or "cuda". If an unsupported device is provided, 
#' the function throws an error.
#'
#' @param device Character string representing the device to be validated. Can be either "cpu" or "cuda".
#'
#' @details
#' This function ensures that the user has provided a valid device type (e.g., for model inference). 
#' If the device is not supported, the function will stop and display an error message.
#'
#' @examples
#' \dontrun{
#' validate_device("cpu")   # Passes
#' validate_device("cuda")  # Passes
#' validate_device("gpu")   # Throws an error
#' }
#'
#' @export
validate_device <- function(device) {
  cat("Validating device:", device, "\n")
  if (!device %in% c("cpu", "cuda")) {
    stop("Invalid device. Choose 'cpu' or 'cuda'.")
  }
  cat("Device validation passed:", device, "\n")
}

#' Softmax Calculation
#'
#' Computes the softmax of a numeric vector. The softmax function converts a vector of values into probabilities
#' that sum to 1.
#'
#' @param x A numeric vector for which the softmax transformation will be applied.
#'
#' @details
#' The softmax function is commonly used in classification problems to convert raw model outputs (logits)
#' into probabilities. The softmax ensures that the resulting probabilities sum to 1, making it suitable
#' for tasks such as multi-class classification.
#'
#' @return A numeric vector of the same length as `x`, where the elements are the softmax probabilities.
#'
#' @examples
#' \dontrun{
#' softmax(c(2, 1, 0))  # Returns probabilities
#' softmax(c(-1, -2, -3))  # Returns probabilities that sum to 1
#' }
#'
#' @export
softmax <- function(x) {
  cat("Calculating softmax for input:", x, "\n")
  exp_x <- exp(x - max(x))
  result <- exp_x / sum(exp_x)
  cat("Softmax result:", result, "\n")
  result
}

#' Remove Stopwords from Text
#'
#' Removes stopwords from a list of input texts. The function compares each word in the input text against a list of 
#' stopwords and removes any matches.
#'
#' @param texts A character vector containing the texts from which stopwords will be removed.
#' @param stopwords A character vector of stopwords to be removed. Typically, this would be provided by 
#' `tm::stopwords("en")`.
#'
#' @details
#' This function processes each text, splits it into individual words, and removes any word that appears in the 
#' provided list of stopwords. It is useful for text preprocessing steps in Natural Language Processing (NLP) pipelines.
#'
#' @return A character vector with the stopwords removed.
#'
#' @examples
#' \dontrun{
#' texts <- c("This is a test", "Text processing is fun")
#' stopwords <- tm::stopwords("en")
#' remove_stopwords(texts, stopwords)
#' }
#'
#' @export
remove_stopwords <- function(texts, stopwords) {
  cat("Removing stopwords from texts...\n")
  result <- sapply(texts, function(text) {
    words <- unlist(strsplit(text, "\\s+"))
    words <- words[!words %in% stopwords]
    paste(words, collapse = " ")
  })
  cat("Text after stopwords removal:", result, "\n")
  result
}
