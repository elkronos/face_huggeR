# 1. Function to write log entries
writeLog <- function(message, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  entry <- paste(timestamp, level, message, sep = " - ")
  cat(entry, file = logFile, append = TRUE, sep = "\n")
  if (verbose) cat(entry, "\n")  # Print to console if verbose mode is on
}

# 2. Function to capture and log session information
logSessionInfo <- function() {
  info <- sessionInfo()
  writeLog(paste("Session Information: ", capture.output(info)), "INFO")
}

# 3. Function to log OS and hardware details
logSystemInfo <- function() {
  sys_info <- Sys.info()
  writeLog(paste("System Information: ", capture.output(sys_info)), "INFO")
  
  # Optional: Fetch CPU, memory, and disk details (OS dependent)
  if (.Platform$OS.type == "unix") {
    cpu_info <- system("lscpu", intern = TRUE)
    writeLog(paste("CPU Information: ", cpu_info), "INFO")
    
    mem_info <- system("free -h", intern = TRUE)
    writeLog(paste("Memory Information: ", mem_info), "INFO")
    
    disk_info <- system("df -h", intern = TRUE)
    writeLog(paste("Disk Information: ", disk_info), "INFO")
  }
}

# 4. Function to log environment variables
logEnvVariables <- function() {
  env_vars <- Sys.getenv()
  writeLog(paste("Environment Variables: ", capture.output(env_vars)), "INFO")
}

# 5. Function to log library paths
logLibPaths <- function() {
  lib_paths <- .libPaths()
  writeLog(paste("Library Paths: ", capture.output(lib_paths)), "INFO")
}

# 6. Function to log installed packages
logPackages <- function() {
  packages <- installed.packages()
  writeLog(paste("Installed Packages: ", capture.output(packages[, c("Package", "Version")])), "INFO")
}

# 7. Function to handle and log errors
logError <- function(e) {
  err_msg <- paste("Error: ", e$message, "\nCall: ", e$call)
  writeLog(err_msg, "ERROR")
  if (onError == "stop") stop(e)  # Stop execution if onError = "stop"
}

# 8. Function to log warnings
logWarnings <- function() {
  warnings_list <- warnings()
  if (!is.null(warnings_list)) {
    writeLog(paste("Warnings: ", capture.output(warnings_list)), "WARN")
  }
}

# 9. Function to log required R packages check
logRequiredPackages <- function() {
  tryCatch({
    writeLog("Checking and loading required R packages", "INFO")
    check_and_load_packages()
    writeLog("All required R packages are installed and loaded", "INFO")
  }, error = function(e) {
    logError(e)
  })
}

# 10. Function to log Python environment setup
logPythonEnvironment <- function(env_name = "r-reticulate", python_version = "3.8", python_packages = c("transformers", "torch")) {
  tryCatch({
    writeLog("Starting Python environment setup", "INFO")
    setup_python_environment(env_name = env_name, python_version = python_version, python_packages = python_packages)
    writeLog("Python environment setup complete", "INFO")
  }, error = function(e) {
    logError(e)
  })
}

# 11. Function to log task validation
logTaskValidation <- function(task, valid_tasks) {
  tryCatch({
    writeLog(paste("Validating task:", task), "INFO")
    validate_task(task, valid_tasks)
    writeLog(paste("Task validated successfully:", task), "INFO")
  }, error = function(e) {
    logError(e)
  })
}

# 12. Function to log device validation
logDeviceValidation <- function(device) {
  tryCatch({
    writeLog(paste("Validating device:", device), "INFO")
    validate_device(device)
    writeLog(paste("Device validated successfully:", device), "INFO")
  }, error = function(e) {
    logError(e)
  })
}

# 13. Function to log API request
logAPIRequest <- function(method, url, body = NULL, retries = 3, pause = 5, api_key = NULL, verbose = TRUE) {
  tryCatch({
    writeLog(paste("Making API request to:", url), "INFO")
    response <- make_api_request(method = method, url = url, body = body, retries = retries, pause = pause, api_key = api_key, verbose = verbose)
    writeLog(paste("API request to", url, "completed successfully."), "INFO")
    return(response)
  }, error = function(e) {
    logError(e)
    return(NULL)
  })
}

# 14. Function to log API rate limit check
logRateLimitCheck <- function(api_key = NULL) {
  tryCatch({
    writeLog("Checking Hugging Face API rate limit", "INFO")
    rate_info <- check_api_rate_limit(api_key)
    if (!is.null(rate_info)) {
      writeLog(paste("Rate limit:", rate_info$limit, "Remaining:", rate_info$remaining, "Resets at:", rate_info$reset), "INFO")
    } else {
      writeLog("Rate limit information unavailable", "WARN")
    }
  }, error = function(e) {
    logError(e)
  })
}

# 15. Function to log API key management
logAPIKeyManagement <- function(api_key, overwrite = FALSE) {
  tryCatch({
    writeLog("Managing Hugging Face API key", "INFO")
    manage_hf_api_key(api_key, overwrite)
    writeLog("API key management completed successfully", "INFO")
  }, error = function(e) {
    logError(e)
  })
}

# 16. Function to log loading of required Python modules
logLoadModules <- function() {
  tryCatch({
    writeLog("Loading required Python modules (transformers, torch)", "INFO")
    modules <- load_modules()
    writeLog("Successfully loaded required Python modules", "INFO")
  }, error = function(e) {
    logError(e)
  })
}

# 17. Function to log loading model and tokenizer
logModelTokenizerLoad <- function(model_name, task, device = "cpu", cache_dir = NULL) {
  tryCatch({
    writeLog(paste("Loading model and tokenizer for model:", model_name, "on device:", device), "INFO")
    model_info <- load_model_tokenizer(model_name, task, device, cache_dir)
    writeLog(paste("Successfully loaded model and tokenizer for model:", model_name), "INFO")
    return(model_info)
  }, error = function(e) {
    logError(e)
    return(NULL)
  })
}

# 18. Function to log CUDA status check
logCudaStatus <- function() {
  tryCatch({
    writeLog("Checking CUDA availability...", "INFO")
    cuda_available <- check_cuda_status()
    if (cuda_available) {
      writeLog("CUDA is available", "INFO")
    } else {
      writeLog("CUDA is not available", "INFO")
    }
  }, error = function(e) {
    logError(e)
  })
}

# 19. Function to log text preprocessing
logPreprocessTexts <- function(texts, lowercase = TRUE, remove_stopwords_flag = FALSE) {
  tryCatch({
    writeLog("Preprocessing texts", "INFO")
    processed_texts <- preprocess_texts(texts, lowercase, remove_stopwords_flag)
    writeLog("Successfully preprocessed texts", "INFO")
    return(processed_texts)
  }, error = function(e) {
    logError(e)
    return(NULL)
  })
}

# 20. Function to log tokenizing texts
logTokenizeTexts <- function(tokenizer, texts, max_length = 128, return_tensors = "pt", padding = TRUE, truncation = TRUE) {
  tryCatch({
    writeLog("Tokenizing texts", "INFO")
    tokenized_inputs <- tokenize_texts(tokenizer, texts, max_length, return_tensors, padding, truncation)
    writeLog("Successfully tokenized texts", "INFO")
    return(tokenized_inputs)
  }, error = function(e) {
    logError(e)
    return(NULL)
  })
}

# 21. Function to log running inference
logRunInference <- function(model, inputs, task, params) {
  tryCatch({
    writeLog(paste("Running inference on task:", task), "INFO")
    result <- run_inference(model, inputs, task, params)
    writeLog("Inference completed successfully", "INFO")
    return(result)
  }, error = function(e) {
    logError(e)
    return(NULL)
  })
}

# 22. Function to log processing outputs
logProcessOutputs <- function(outputs, tokenizer, task, inputs = NULL) {
  tryCatch({
    writeLog(paste("Processing outputs for task:", task), "INFO")
    processed_output <- process_outputs(outputs, tokenizer, task, inputs)
    writeLog("Successfully processed outputs", "INFO")
    return(processed_output)
  }, error = function(e) {
    logError(e)
    return(NULL)
  })
}

# 23. Main function to generate a log file
generateLog <- function(outputFile = logFile, verboseMode = TRUE, errorHandling = "continue") {
  # Set parameters
  logFile <<- outputFile
  verbose <<- verboseMode
  onError <<- errorHandling
  
  writeLog("Starting system log generation...", "INFO")
  
  tryCatch({
    logSessionInfo()
    logSystemInfo()
    logEnvVariables()
    logLibPaths()
    logPackages()
    logWarnings()
  }, error = function(e) {
    logError(e)
  })
  
  writeLog("System log generation completed.", "INFO")
}

# 24. Extended master log generator with inference
generateExtendedLogWithInference <- function(
    model_name = "distilbert-base-uncased", 
    task = "classification", 
    texts = c("This is a test sentence."), 
    device = "cpu", 
    cache_dir = NULL,
    outputFile = logFile, 
    verboseMode = TRUE, 
    errorHandling = "continue"
) {
  # Set parameters
  logFile <<- outputFile
  verbose <<- verboseMode
  onError <<- errorHandling
  
  writeLog("Starting extended system and environment log generation with inference...", "INFO")
  
  tryCatch({
    # Standard system information
    logSessionInfo()
    logSystemInfo()
    logEnvVariables()
    logLibPaths()
    logPackages()
    
    # Python environment setup and logging
    logRequiredPackages()
    logPythonEnvironment()  # Log Python environment setup
    
    # Validate tasks and devices
    logTaskValidation(task, c("classification", "generation"))
    logDeviceValidation(device)
    
    # Log loading of modules, model, and tokenizer
    logLoadModules()
    model_info <- logModelTokenizerLoad(model_name, task, device, cache_dir)
    
    # Log CUDA status
    logCudaStatus()
    
    # Preprocess texts
    processed_texts <- logPreprocessTexts(texts, lowercase = TRUE, remove_stopwords_flag = FALSE)
    
    # Tokenize texts
    tokenized_inputs <- logTokenizeTexts(model_info$tokenizer, processed_texts)
    
    # Run inference and process outputs
    inference_result <- logRunInference(model_info$model, tokenized_inputs, task, list(max_length = 128))
    processed_output <- logProcessOutputs(inference_result, model_info$tokenizer, task, tokenized_inputs)
    
    writeLog("Extended log generation with model inference completed successfully.", "INFO")
    
    # Return processed outputs for further analysis if needed
    return(processed_output)
    
  }, error = function(e) {
    logError(e)
  })
}
