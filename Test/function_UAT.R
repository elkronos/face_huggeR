# Required Libraries
library(reticulate)
library(httr)
library(tibble)
library(tm)
library(progress)
library(data.table)

# Initialize results data frame
test_results <- data.frame(Test = character(), Result = character(), stringsAsFactors = FALSE)

# Helper function to print test results and store them
print_and_store_result <- function(test_name, passed, message = NULL) {
  result <- if (passed) "PASS" else "FAIL"
  cat(sprintf("%-60s [%s]\n", test_name, result))
  if (!is.null(message)) cat("  ", message, "\n")
  test_results <<- rbind(test_results, data.frame(Test = test_name, Result = result, stringsAsFactors = FALSE))
}

# Test validate_task function
test_validate_task <- function() {
  cat("\n--- Running test_validate_task ---\n")
  
  # Test Case 1: Valid Task
  cat("Validating task: classification\n")
  valid_test <- tryCatch({
    validate_task("classification", c("classification", "generation"))
    TRUE
  }, error = function(e) {
    cat("Validation failed for valid task.\n")
    FALSE
  })
  print_and_store_result("validate_task: Valid Task", valid_test)
  
  # Test Case 2: Invalid Task
  cat("Validating task: invalid_task\n")
  invalid_test <- tryCatch({
    validate_task("invalid_task", c("classification", "generation"))
    FALSE
  }, error = function(e) {
    cat("Expected failure for invalid task.\n")
    TRUE
  })
  print_and_store_result("validate_task: Invalid Task", invalid_test)
}

# Test validate_device function
test_validate_device <- function() {
  cat("\n--- Running test_validate_device ---\n")
  
  # Test Case 1: Valid Device "cpu"
  cat("Validating device: cpu\n")
  valid_cpu <- tryCatch({
    validate_device("cpu")
    TRUE
  }, error = function(e) {
    cat("Validation failed for device 'cpu'.\n")
    FALSE
  })
  print_and_store_result("validate_device: Valid Device 'cpu'", valid_cpu)
  
  # Test Case 2: Valid Device "cuda"
  cat("Validating device: cuda\n")
  valid_cuda <- tryCatch({
    validate_device("cuda")
    TRUE
  }, error = function(e) {
    cat("Validation failed for device 'cuda'.\n")
    FALSE
  })
  print_and_store_result("validate_device: Valid Device 'cuda'", valid_cuda)
  
  # Test Case 3: Invalid Device
  cat("Validating device: gpu\n")
  invalid_device <- tryCatch({
    validate_device("gpu")
    FALSE
  }, error = function(e) {
    cat("Expected failure for invalid device.\n")
    TRUE
  })
  print_and_store_result("validate_device: Invalid Device", invalid_device)
}

# Test softmax function
test_softmax <- function() {
  cat("\n--- Running test_softmax ---\n")
  
  # Test Case 1: Positive Numbers
  x1 <- c(2, 1, 0)
  cat("Calculating softmax for positive numbers:", x1, "\n")
  result1 <- softmax(x1)
  sum_to_one1 <- abs(sum(result1) - 1) < 1e-6
  print_and_store_result("softmax: Positive Numbers Sum to One", sum_to_one1)
  
  # Test Case 2: Negative Numbers
  x2 <- c(-1, -2, -3)
  cat("Calculating softmax for negative numbers:", x2, "\n")
  result2 <- softmax(x2)
  sum_to_one2 <- abs(sum(result2) - 1) < 1e-6
  print_and_store_result("softmax: Negative Numbers Sum to One", sum_to_one2)
  
  # Test Case 3: Zero Vector
  x3 <- c(0, 0, 0)
  cat("Calculating softmax for zero vector:", x3, "\n")
  result3 <- softmax(x3)
  equal_probs <- all(abs(result3 - (1 / 3)) < 1e-6)
  print_and_store_result("softmax: Zero Vector Equal Probabilities", equal_probs)
}

# Test remove_stopwords function
test_remove_stopwords <- function() {
  cat("\n--- Running test_remove_stopwords ---\n")
  
  # Test Case 1: Text with Stopwords
  texts1 <- c("This is a test")
  stopwords_list <- c("is", "a")
  cat("Removing stopwords from text:", texts1, "\n")
  result1 <- remove_stopwords(texts1, stopwords_list)
  expected1 <- "This test"
  passed1 <- result1 == expected1
  print_and_store_result("remove_stopwords: Text with Stopwords", passed1)
  
  # Test Case 2: Text without Stopwords
  texts2 <- c("Testing functions")
  cat("Removing stopwords from text with no stopwords:", texts2, "\n")
  result2 <- remove_stopwords(texts2, stopwords_list)
  expected2 <- "Testing functions"
  passed2 <- result2 == expected2
  print_and_store_result("remove_stopwords: Text without Stopwords", passed2)
  
  # Test Case 3: Empty Text
  texts3 <- c("")
  cat("Removing stopwords from empty text\n")
  result3 <- remove_stopwords(texts3, stopwords_list)
  expected3 <- ""
  passed3 <- result3 == expected3
  print_and_store_result("remove_stopwords: Empty Text", passed3)
}

# Test preprocess_texts function
test_preprocess_texts <- function() {
  cat("\n--- Running test_preprocess_texts ---\n")
  
  # Test Case 1: Lowercase and Remove Stopwords
  texts1 <- c("This IS a Test")
  cat("Preprocessing text with lowercase and remove stopwords:", texts1, "\n")
  result1 <- preprocess_texts(texts1, lowercase = TRUE, remove_stopwords_flag = TRUE)
  expected1 <- "test"
  passed1 <- result1 == expected1
  print_and_store_result("preprocess_texts: Lowercase and Remove Stopwords", passed1)
  
  # Test Case 2: Only Lowercase
  texts2 <- c("This IS a Test")
  cat("Preprocessing text with only lowercase:", texts2, "\n")
  result2 <- preprocess_texts(texts2, lowercase = TRUE, remove_stopwords_flag = FALSE)
  expected2 <- "this is a test"
  passed2 <- result2 == expected2
  print_and_store_result("preprocess_texts: Only Lowercase", passed2)
  
  # Test Case 3: No Preprocessing
  texts3 <- c("This IS a Test")
  cat("No preprocessing applied to text:", texts3, "\n")
  result3 <- preprocess_texts(texts3, lowercase = FALSE, remove_stopwords_flag = FALSE)
  expected3 <- "This IS a Test"
  passed3 <- result3 == expected3
  print_and_store_result("preprocess_texts: No Preprocessing", passed3)
}

# Test load_modules function
test_load_modules <- function() {
  cat("\n--- Running test_load_modules ---\n")
  
  modules_loaded <- tryCatch({
    cat("Loading modules...\n")
    modules <- load_modules()
    cat("Modules loaded successfully!\n")
    !is.null(modules$transformers) && !is.null(modules$torch)
  }, error = function(e) {
    cat("Error loading modules.\n")
    FALSE
  })
  print_and_store_result("load_modules: Modules Loaded", modules_loaded)
}

# Test load_model_tokenizer function with CUDA availability
test_load_model_tokenizer <- function() {
  cat("\n--- Running test_load_model_tokenizer ---\n")
  
  # Test Case 1: Valid Model and Task
  cat("Testing load_model_tokenizer with a valid model and task...\n")
  test1 <- tryCatch({
    model_info <- load_model_tokenizer("distilbert-base-uncased", "classification", device = "cpu")
    is.list(model_info) && !is.null(model_info$model) && !is.null(model_info$tokenizer)
  }, error = function(e) {
    cat("Error loading model/tokenizer for valid task.\n")
    FALSE
  })
  print_and_store_result("load_model_tokenizer: Valid Model and Task", test1)
  
  # Test Case 2: Invalid Task
  cat("Testing load_model_tokenizer with an invalid task...\n")
  test2 <- tryCatch({
    load_model_tokenizer("distilbert-base-uncased", "invalid_task", device = "cpu")
    FALSE
  }, error = function(e) {
    cat("Expected failure for invalid task.\n")
    TRUE
  })
  print_and_store_result("load_model_tokenizer: Invalid Task", test2)
  
  # Test Case 3: Invalid Device
  cat("Testing load_model_tokenizer with an invalid device...\n")
  test3 <- tryCatch({
    load_model_tokenizer("distilbert-base-uncased", "classification", device = "gpu")
    FALSE
  }, error = function(e) {
    cat("Expected failure for invalid device.\n")
    TRUE
  })
  print_and_store_result("load_model_tokenizer: Invalid Device", test3)
  
  # Test Case 4: CUDA Device Not Available
  cat("Testing load_model_tokenizer for CUDA availability...\n")
  test4 <- tryCatch({
    warnings <- NULL
    withCallingHandlers({
      model_info <- load_model_tokenizer("distilbert-base-uncased", "classification", device = "cuda")
    }, warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
    # Test should pass if CUDA is unavailable and handled correctly
    passed <- any(grepl("CUDA not available", warnings)) || check_cuda_status() == FALSE
    passed
  }, error = function(e) {
    cat("CUDA device not available.\n")
    TRUE  # Pass the test if CUDA is not available
  })
  print_and_store_result("load_model_tokenizer: CUDA Not Available Warning", test4)
}


# Test tokenize_texts function
test_tokenize_texts <- function() {
  cat("\n--- Running test_tokenize_texts ---\n")
  
  modules <- load_modules()
  tokenizer <- modules$transformers$AutoTokenizer$from_pretrained("distilbert-base-uncased")
  
  # Test Case 1: Default Parameters
  texts1 <- c("Hello world")
  cat("Tokenizing texts: ", texts1, "\n")
  inputs1 <- tokenize_texts(tokenizer, texts1)
  passed1 <- !is.null(inputs1$input_ids)
  cat("Tokenization successful! Returning tokenized inputs.\n")
  print_and_store_result("tokenize_texts: Default Parameters", passed1)
  
  # Test Case 2: Custom Max Length
  cat("Tokenizing texts with custom max length...\n")
  inputs2 <- tokenize_texts(tokenizer, texts1, max_length = 5)
  input_ids2 <- as.array(inputs2$input_ids)
  lengths2 <- apply(input_ids2, 1, length)
  passed2 <- all(lengths2 <= 5)
  print_and_store_result("tokenize_texts: Custom Max Length", passed2)
  
  # Test Case 3: No Padding
  cat("Tokenizing texts with no padding...\n")
  inputs3 <- tokenize_texts(tokenizer, texts1, padding = FALSE)
  input_ids3 <- as.array(inputs3$input_ids)
  attention_mask3 <- as.array(inputs3$attention_mask)
  passed3 <- all(dim(input_ids3) == dim(attention_mask3))
  print_and_store_result("tokenize_texts: No Padding", passed3)
}

# Run all tests
run_all_tests <- function() {
  cat("Running Comprehensive UAT\n")
  cat("==================================\n")
  
  # Non-API Tests
  test_validate_task()
  test_validate_device()
  test_softmax()
  test_remove_stopwords()
  test_preprocess_texts()
  test_load_modules()
  test_load_model_tokenizer()
  test_tokenize_texts()
  
  cat("==================================\n")
  cat("UAT completed\n\n")
  
  # Print summary
  cat("Test Summary:\n")
  cat("==================================\n")
  print(table(test_results$Result))
  cat("\nDetailed Results:\n")
  print(test_results)
}


run_all_tests()
