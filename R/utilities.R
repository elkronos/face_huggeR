################################################################################ Packages
required_packages <- c("tm", "stringr",
                       "reticulate", "tibble", "progress", "tm", "httr", "data.table")

# Function to check if a package is installed, and install if missing
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

################################################################################ Local helpers

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


################################################################################ API helpers

#' Retrieve API Key
#'
#' Retrieves the API key from the environment or uses the provided key. If no API key is passed as an argument, 
#' the function checks for the `HF_API_KEY` environment variable. If neither is available, an error is thrown.
#'
#' @param api_key A character string containing the API key. If `NULL` (default), the function will attempt to retrieve 
#' the key from the `HF_API_KEY` environment variable.
#'
#' @details
#' This function is designed to fetch an API key for use in functions that require authentication. It first checks 
#' if the key is provided directly via the `api_key` argument. If not, it attempts to retrieve the API key from the 
#' environment variable `HF_API_KEY`. If the environment variable is also not set, the function stops with an error 
#' message.
#'
#' @return A character string containing the API key.
#'
#' @examples
#' \dontrun{
#' # Retrieve API key from environment variable
#' api_key <- get_api_key()
#' 
#' # Provide API key directly
#' api_key <- get_api_key("my_secret_api_key")
#' }
#'
#' @export
get_api_key <- function(api_key = NULL) {
  if (is.null(api_key)) {
    api_key <- Sys.getenv("HF_API_KEY")
    if (api_key == "") {
      stop("API key is not provided and 'HF_API_KEY' environment variable is not set.")
    }
  }
  return(api_key)
}

#' Make an API Request with Retry Mechanism
#'
#' Sends an API request using the specified HTTP method, URL, and optional request body. Includes a retry mechanism
#' with exponential backoff in case of failures or rate limit errors (status code 429).
#'
#' @param method A character string representing the HTTP method to be used (e.g., "GET", "POST", "PUT", etc.).
#' @param url A character string specifying the URL endpoint for the API request.
#' @param body Optional request body, typically provided as a list or JSON-like structure. Default is `NULL`.
#' @param retries An integer indicating the number of retry attempts if the request fails. Default is `3`.
#' @param pause A numeric value indicating the initial pause duration (in seconds) between retries. Default is `5`.
#' @param api_key Optional character string for the API key. If `NULL` (default), the API key will be fetched using 
#' the `get_api_key` function.
#' @param verbose Logical flag indicating whether detailed messages should be printed during the request process. Default is `TRUE`.
#'
#' @details
#' This function sends an HTTP request using the provided method and URL. If the request fails or if a rate limit 
#' (status 429) is encountered, it will retry the request up to `retries` times, with an exponential backoff delay 
#' between attempts. The API key is passed in the `Authorization` header as a Bearer token.
#' 
#' If the response status is 200 and the content is successfully parsed, the parsed response content is returned. 
#' For status codes other than 200 and 429, the function logs the response status and error message, if available.
#'
#' @return The parsed response content if the request is successful. If all retry attempts fail, the function will
#' stop with an error.
#'
#' @examples
#' \dontrun{
#' # Make a GET request to an API endpoint
#' response <- make_api_request(
#'   method = "GET",
#'   url = "https://api.example.com/data",
#'   api_key = "your_api_key_here"
#' )
#' 
#' # Make a POST request with a request body
#' response <- make_api_request(
#'   method = "POST",
#'   url = "https://api.example.com/upload",
#'   body = list(name = "example", type = "data"),
#'   api_key = "your_api_key_here"
#' )
#' }
#'
#' @export
make_api_request <- function(
    method,
    url,
    body = NULL,
    retries = 3,
    pause = 5,
    api_key = NULL,
    verbose = TRUE
) {
  api_key <- get_api_key(api_key)
  
  for (attempt in seq_len(retries)) {
    tryCatch({
      response <- httr::VERB(
        method,
        url,
        httr::add_headers(`Authorization` = paste("Bearer", api_key)),
        body = body,
        encode = "json",
        httr::timeout(60)
      )
      
      if (response$status_code == 200) {
        response_content <- httr::content(response, as = "parsed", simplifyVector = TRUE)
        if (!is.null(response_content$error)) {
          if (verbose) message("Error in response: ", response_content$error)
          next
        }
        return(response_content)
      } else if (response$status_code == 429) {
        # Rate limit exceeded
        if (verbose) message("Rate limit exceeded. Waiting before retrying...")
        retry_after <- as.numeric(httr::headers(response)$`retry-after`)
        if (is.na(retry_after)) retry_after <- pause * 2^(attempt - 1)
        Sys.sleep(retry_after)
      } else {
        error_message <- httr::content(response, as = "text", encoding = "UTF-8")
        if (verbose) message(
          sprintf(
            "Attempt %d failed with status %d: %s",
            attempt,
            response$status_code,
            error_message
          )
        )
      }
    }, error = function(e) {
      if (verbose) message(sprintf("Error on attempt %d: %s. Retrying...", attempt, e$message))
    })
    Sys.sleep(pause * 2^(attempt - 1))  # Exponential backoff
  }
  stop("Failed after ", retries, " retries.")
}

#' Exponential Backoff with Maximum Wait Time
#'
#' Implements an exponential backoff mechanism for delaying execution. The wait time increases exponentially 
#' with each retry attempt, up to a maximum pause duration.
#'
#' @param attempt An integer representing the current retry attempt. The wait time increases exponentially with the attempt number.
#' @param base_pause A numeric value specifying the base wait time (in seconds) for the first retry attempt. Default is `5` seconds.
#' @param max_pause A numeric value indicating the maximum wait time (in seconds) allowed between attempts. Default is `60` seconds.
#'
#' @details
#' The exponential backoff algorithm increases the wait time by powers of two with each successive retry attempt, starting 
#' from the `base_pause` duration. The wait time is capped at `max_pause` seconds to prevent excessively long delays.
#' This is useful for handling retries in scenarios such as rate limiting or transient errors in API requests.
#'
#' @return This function does not return a value. It pauses execution for the calculated wait time.
#'
#' @examples
#' \dontrun{
#' # Wait for 5 seconds on the first attempt, 10 seconds on the second, and so on.
#' exponential_backoff(attempt = 1)
#' exponential_backoff(attempt = 2)
#' 
#' # Use a different base pause and max pause
#' exponential_backoff(attempt = 3, base_pause = 2, max_pause = 30)
#' }
#'
#' @export
exponential_backoff <- function(attempt, base_pause = 5, max_pause = 60) {
  wait_time <- min(base_pause * 2^(attempt - 1), max_pause)
  Sys.sleep(wait_time)
}

#' Manage Hugging Face API Key
#'
#' Sets or updates the Hugging Face API key in the environment. If a key is already set, the function allows 
#' for optional overwriting based on user input or the `overwrite` argument.
#'
#' @param api_key A character string representing the new API key to be set for Hugging Face.
#' @param overwrite A logical flag indicating whether to overwrite the existing API key without prompting the user. 
#' Default is `FALSE`. If `FALSE` and an API key is already set, the function will prompt the user for confirmation 
#' in an interactive session.
#'
#' @details
#' This function checks if an API key is already set in the `HF_API_KEY` environment variable. If an existing key is 
#' found and `overwrite` is `FALSE`, it will prompt the user for confirmation (in an interactive session) before 
#' overwriting the key. In non-interactive sessions, it will not change the key unless `overwrite` is `TRUE`.
#'
#' Once the API key is set, it is stored in the environment variable `HF_API_KEY`.
#'
#' @return Returns `TRUE` if the API key was successfully set or updated, and `FALSE` if the key was not changed. 
#' The return value is invisible.
#'
#' @examples
#' \dontrun{
#' # Set a new API key
#' manage_hf_api_key("your_new_api_key")
#' 
#' # Set a new API key and overwrite the existing one without prompting
#' manage_hf_api_key("your_new_api_key", overwrite = TRUE)
#' }
#'
#' @export
manage_hf_api_key <- function(api_key, overwrite = FALSE) {
  existing_key <- Sys.getenv("HF_API_KEY")
  if (existing_key != "" && !overwrite) {
    if (interactive()) {
      overwrite_prompt <- readline(prompt = "An API key is already set. Overwrite? (y/n): ")
      if (tolower(overwrite_prompt) != "y") {
        message("API key not changed.")
        return(invisible(FALSE))
      }
    } else {
      message("API key is already set and 'overwrite' is FALSE. API key not changed.")
      return(invisible(FALSE))
    }
  }
  Sys.setenv(HF_API_KEY = api_key)
  message("Hugging Face API key set.")
  invisible(TRUE)
}

#' Check API Rate Limit for Hugging Face
#'
#' Retrieves the current API rate limit status for a Hugging Face model API endpoint. The function sends a `HEAD` request 
#' to check the rate limit headers and returns the limit, remaining requests, and reset time.
#'
#' @param api_key A character string representing the Hugging Face API key. If `NULL`, the API key will be retrieved using 
#' the `get_api_key` function from the environment variable `HF_API_KEY`.
#'
#' @details
#' This function sends a `HEAD` request to the Hugging Face model endpoint and checks for the presence of rate limit headers 
#' (`x-ratelimit-limit`, `x-ratelimit-remaining`, and `x-ratelimit-reset`). If found, it returns a list with the rate limit, 
#' the number of remaining requests, and the time at which the limit will reset. If the headers are not present or if the 
#' request fails, it will return `NULL` and display an appropriate message.
#'
#' @return A list containing three elements:
#' \describe{
#'   \item{limit}{The maximum number of requests allowed.}
#'   \item{remaining}{The number of requests remaining before the limit is reached.}
#'   \item{reset}{The time at which the rate limit will reset (in UTC).}
#' }
#' Returns `NULL` if the rate limit headers are not found or if the request fails.
#'
#' @examples
#' \dontrun{
#' # Check the API rate limit using an existing API key
#' rate_limit <- check_api_rate_limit()
#' 
#' if (!is.null(rate_limit)) {
#'   print(rate_limit)
#' }
#' }
#'
#' @export
check_api_rate_limit <- function(api_key = NULL) {
  api_key <- get_api_key(api_key)
  url <- "https://api-inference.huggingface.co/models/distilbert-base-uncased"
  
  tryCatch({
    response <- httr::HEAD(
      url,
      httr::add_headers(`Authorization` = paste("Bearer", api_key)),
      httr::timeout(30)
    )
    if (response$status_code == 200) {
      headers <- httr::headers(response)
      if (!is.null(headers$`x-ratelimit-limit`)) {
        rate_limit <- list(
          limit = as.numeric(headers$`x-ratelimit-limit`),
          remaining = as.numeric(headers$`x-ratelimit-remaining`),
          reset = as.POSIXct(
            as.numeric(headers$`x-ratelimit-reset`),
            origin = "1970-01-01",
            tz = "UTC"
          )
        )
        return(rate_limit)
      } else {
        message("Rate limit headers not found in the response.")
        return(NULL)
      }
    } else {
      message(sprintf("Failed to retrieve rate limit info. Status: %d", response$status_code))
      return(NULL)
    }
  }, error = function(e) {
    message(sprintf("Error: %s", e$message))
    return(NULL)
  })
}

#' Check and Wait for API Rate Limit Reset
#'
#' Checks the current API rate limit status and pauses execution if the limit has been reached. The function waits until the rate limit resets before proceeding.
#'
#' @param api_key A character string representing the Hugging Face API key. If `NULL`, the API key will be retrieved using 
#' the `get_api_key` function from the environment variable `HF_API_KEY`.
#'
#' @details
#' This function first calls `check_api_rate_limit` to retrieve the current rate limit status. If the remaining number of 
#' requests is zero, the function calculates the time until the rate limit reset and waits for that duration before proceeding.
#' If the rate limit reset time has already passed or the rate limit information is unavailable, the function proceeds without waiting.
#'
#' @return This function does not return a value. It pauses execution if the rate limit has been reached and waits until it resets.
#'
#' @examples
#' \dontrun{
#' # Check and wait for rate limit reset if necessary
#' check_and_wait_rate_limit()
#' 
#' # Use an explicit API key
#' check_and_wait_rate_limit(api_key = "your_api_key_here")
#' }
#'
#' @export
check_and_wait_rate_limit <- function(api_key = NULL) {
  rate_info <- check_api_rate_limit(api_key)
  if (!is.null(rate_info) && !is.na(rate_info$remaining) && rate_info$remaining == 0) {
    wait_time <- as.numeric(difftime(rate_info$reset, Sys.time(), units = "secs"))
    if (wait_time > 0) {
      message("Rate limit hit. Waiting for ", round(wait_time, 2), " seconds...")
      Sys.sleep(wait_time)
    } else {
      message("Rate limit reset time has passed or invalid. Proceeding without wait.")
    }
  } else if (is.null(rate_info)) {
    message("Unable to retrieve rate limit information. Proceeding cautiously.")
  }
}
