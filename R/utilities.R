required_packages <- c("tm", "stringr")

# Function to check if a package is installed, and install if missing
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}


#' Validate Task
#'
#' Validates if the provided task is supported.
#'
#' @param task Character string specifying the task to validate.
#' @param valid_tasks A character vector of valid tasks.
#'
#' @return None. Stops execution if the task is invalid.
#'
#' @examples
#' \dontrun{
#' validate_task("classification", c("classification", "generation"))
#' }
#'
#' @export
validate_task <- function(task, valid_tasks) {
  cat("Validating task:", task, "\n")
  if (!(task %in% valid_tasks)) {
    cat("Invalid task specified:", task, "\n")
    stop("Invalid task specified.")
  }
  cat("Task validation passed:", task, "\n")
}


#' Validate Device
#'
#' Validates if the provided device is supported.
#'
#' @param device Character string specifying the device to validate.
#'
#' @return None. Stops execution if the device is invalid.
#'
#' @examples
#' \dontrun{
#' validate_device("cpu")
#' }
#'
#' @export
validate_device <- function(device) {
  cat("Validating device:", device, "\n")
  valid_devices <- c("cpu", "cuda")
  if (!(device %in% valid_devices)) {
    cat("Invalid device specified:", device, "\n")
    stop("Invalid device specified.")
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
