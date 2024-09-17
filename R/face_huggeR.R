
################################################################################ Local

#' Load Required Modules for Model and Tokenizer
#'
#' This function loads the necessary Python modules for running Hugging Face models and PyTorch. 
#' Specifically, it loads the `transformers` and `torch` libraries using `reticulate` for interfacing 
#' between R and Python.
#'
#' @return A list of loaded modules (`transformers`, `torch`). If any module fails to load, the function 
#' stops with an error.
#'
#' @details
#' This function is essential for working with Hugging Face models in R through the `reticulate` package. 
#' It uses delayed loading for efficiency and handles errors by providing informative messages.
#'
#' @import reticulate
#' @examples
#' \dontrun{
#' modules <- load_modules()
#' print(modules$transformers)
#' print(modules$torch)
#' }
#' 
#' @export
load_modules <- function() {
  cat("Loading required modules...\n")
  
  modules <- list(
    transformers = tryCatch({
      reticulate::import("transformers", delay_load = TRUE)
    }, error = function(e) {
      cat("Error loading 'transformers' module:", conditionMessage(e), "\n")
      stop("Failed to load transformers module.")
    }),
    torch = tryCatch({
      reticulate::import("torch", delay_load = TRUE)
    }, error = function(e) {
      cat("Error loading 'torch' module:", conditionMessage(e), "\n")
      stop("Failed to load torch module.")
    })
  )
  
  cat("Modules loaded successfully!\n")
  modules
}

#' Load Model and Tokenizer
#'
#' This function loads a Hugging Face model and tokenizer into the specified device (CPU or CUDA) from the Hugging Face
#' model hub or from a local cache. It automatically sets up the environment for running the model in R.
#'
#' @param model_name Character string specifying the name of the model to load from the Hugging Face model hub.
#' @param task Character string specifying the task type (e.g., "classification", "generation"). 
#' @param device Character string specifying the device to run the model on. Can be "cpu" or "cuda". Default is "cpu".
#' @param cache_dir Character string specifying the directory to cache the model. If `NULL`, a default directory is used.
#'
#' @return A list with two elements: the `model` and the `tokenizer`.
#'
#' @details
#' This function supports several common NLP tasks (classification, generation, token classification, question-answering, 
#' summarization) and handles the loading of both the model and tokenizer. It checks for CUDA availability if a GPU device 
#' is selected and automatically moves the model to CUDA if available.
#'
#' @import reticulate
#' @importFrom utils dir.create
#' @examples
#' \dontrun{
#' model_info <- load_model_tokenizer("distilbert-base-uncased", "classification", device = "cpu")
#' print(model_info$model)
#' print(model_info$tokenizer)
#' }
#'
#' @export
load_model_tokenizer <- function(model_name, task, device = "cpu", cache_dir = NULL) {
  modules <- load_modules()
  transformers <- modules$transformers
  
  cat("Initial cache_dir value:", cache_dir, "\n")
  
  if (is.null(cache_dir)) {
    cache_dir <- normalizePath("~/.cache/huggingface/", mustWork = FALSE)
    cat("Using default cache directory:", cache_dir, "\n")
  } else {
    cache_dir <- normalizePath(cache_dir, mustWork = FALSE)
    cat("Using specified cache directory:", cache_dir, "\n")
  }
  
  if (!dir.exists(cache_dir)) {
    cat("Cache directory does not exist. Creating it...\n")
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  task_map <- list(
    "classification"       = "AutoModelForSequenceClassification",
    "generation"           = "AutoModelForCausalLM",
    "token-classification" = "AutoModelForTokenClassification",
    "question-answering"   = "AutoModelForQuestionAnswering",
    "summarization"        = "AutoModelForSeq2SeqLM"
  )
  
  validate_task(task, names(task_map))
  validate_device(device)
  
  model_class <- transformers[[task_map[[task]]]]
  
  Sys.setenv(HF_HUB_DISABLE_SYMLINKS_WARNING = "1")
  
  model <- tryCatch({
    model_class$from_pretrained(model_name, cache_dir = cache_dir)
  }, error = function(e) {
    cat("Error loading model from cache or remote:\n")
    py_err <- reticulate::py_last_error()
    if (!is.null(py_err)) {
      cat(py_err$type, ":", py_err$value, "\n")
      cat("Traceback:\n", paste(py_err$traceback, collapse = "\n"), "\n")
    } else {
      cat("No Python error available.\n")
    }
    stop("Failed to load model.")
  })
  
  tokenizer <- tryCatch({
    transformers$AutoTokenizer$from_pretrained(model_name, cache_dir = cache_dir)
  }, error = function(e) {
    cat("Error loading tokenizer:\n")
    py_err <- reticulate::py_last_error()
    if (!is.null(py_err)) {
      cat(py_err$type, ":", py_err$value, "\n")
      cat("Traceback:\n", paste(py_err$traceback, collapse = "\n"), "\n")
    } else {
      cat("No Python error available.\n")
    }
    stop("Failed to load tokenizer.")
  })
  
  if (device == "cuda" && check_cuda_status()) {
    cat("Moving model to CUDA device...\n")
    model$to(modules$torch$device("cuda"))
  }
  
  list(model = model, tokenizer = tokenizer)
}

#' Check CUDA Availability
#'
#' This function checks if CUDA is available on the system, and if available, returns the number of available GPUs.
#'
#' @return A logical value indicating whether CUDA is available (`TRUE`) or not (`FALSE`).
#'
#' @details
#' The function loads the `torch` module via `reticulate` and checks if CUDA is available on the system. If available, 
#' it prints the number of CUDA devices.
#'
#' @examples
#' \dontrun{
#' check_cuda_status()
#' }
#'
#' @export
check_cuda_status <- function() {
  cat("Checking CUDA availability...\n")
  modules <- load_modules()
  torch <- modules$torch
  
  if (torch$cuda$is_available()) {
    cat("CUDA is available. Number of available CUDA GPUs:", torch$cuda$device_count(), "\n")
    return(TRUE)
  } else {
    cat("CUDA is not available on this system.\n")
    return(FALSE)
  }
}


#' Preprocess Text for NLP
#'
#' This function preprocesses a vector of texts by optionally lowercasing them and removing stopwords.
#'
#' @param texts A character vector of texts to preprocess.
#' @param lowercase Logical indicating whether to convert text to lowercase. Default is `TRUE`.
#' @param remove_stopwords_flag Logical indicating whether to remove stopwords. Default is `FALSE`.
#'
#' @return A character vector of preprocessed texts.
#'
#' @details
#' This function handles basic text preprocessing for NLP tasks. It includes converting text to lowercase and 
#' removing stopwords using the `tm` package. These preprocessing steps are common in NLP pipelines before 
#' tokenization or model input.
#'
#' @import tm
#' @examples
#' \dontrun{
#' texts <- c("This is an example sentence.")
#' preprocessed_texts <- preprocess_texts(texts, lowercase = TRUE, remove_stopwords_flag = TRUE)
#' print(preprocessed_texts)
#' }
#'
#' @export
preprocess_texts <- function(texts, lowercase = TRUE, remove_stopwords_flag = FALSE) {
  cat("Preprocessing texts...\n")
  if (lowercase) {
    texts <- tolower(texts)
    cat("Lowercased texts:", texts, "\n")
  }
  if (remove_stopwords_flag) {
    stopwords <- tm::stopwords("en")
    texts <- remove_stopwords(texts, stopwords)
  }
  cat("Preprocessed texts:", texts, "\n")
  texts
}

#' Tokenize Texts for NLP Models
#'
#' This function tokenizes a character vector of texts using a specified tokenizer, and returns input tensors 
#' for use with NLP models.
#'
#' @param tokenizer A tokenizer object from the `transformers` library.
#' @param texts A character vector of texts to tokenize.
#' @param max_length Integer specifying the maximum length of the tokenized sequences. Default is `128`.
#' @param return_tensors Character string specifying the format of returned tensors. Default is `"pt"` (PyTorch tensors).
#' @param padding Logical indicating whether to pad the sequences to the maximum length. Default is `TRUE`.
#' @param truncation Logical indicating whether to truncate the sequences to the maximum length. Default is `TRUE`.
#'
#' @return A list containing `input_ids` and `attention_mask` arrays, which are the tokenized inputs for the model.
#'
#' @details
#' This function converts a character vector of texts into tokenized inputs for use in Hugging Face models. 
#' The `input_ids` and `attention_mask` arrays are returned as PyTorch tensors by default.
#'
#' @import reticulate
#' @examples
#' \dontrun{
#' tokenizer <- load_modules()$transformers$AutoTokenizer$from_pretrained("distilbert-base-uncased")
#' tokenized_texts <- tokenize_texts(tokenizer, c("Hello, world!"))
#' print(tokenized_texts)
#' }
#'
#' @export
tokenize_texts <- function(tokenizer, texts, max_length = 128, return_tensors = "pt", padding = TRUE, truncation = TRUE) {
  cat("Tokenizing texts:", texts, "\n")
  
  inputs <- tokenizer(
    texts,
    return_tensors = return_tensors,
    max_length     = as.integer(max_length),
    padding        = padding,
    truncation     = truncation
  )
  
  inputs_r <- reticulate::py_to_r(inputs)
  
  input_ids <- as.array(inputs_r$input_ids$detach()$cpu()$numpy())
  attention_mask <- as.array(inputs_r$attention_mask$detach()$cpu()$numpy())
  
  cat("Tokenized input_ids dimensions:", dim(input_ids), "\n")
  cat("Tokenized attention_mask dimensions:", dim(attention_mask), "\n")
  
  list(input_ids = input_ids, attention_mask = attention_mask)
}

#' Run Model Inference on Task
#'
#' This function runs inference on a specified NLP task (e.g., classification, generation) using a pre-trained model.
#'
#' @param model The Hugging Face model object to run inference with.
#' @param inputs A list of inputs (e.g., tokenized texts) for the model.
#' @param task A character string specifying the task (e.g., "generation", "classification").
#' @param params A list of additional parameters for inference, such as `max_length`, `temperature`, and `num_beams`.
#'
#' @return The model output after inference.
#'
#' @details
#' This function supports multiple NLP tasks. Depending on the task type, it generates or classifies text using
#' the model and returns the output for further processing.
#'
#' @examples
#' \dontrun{
#' model_info <- load_model_tokenizer("distilbert-base-uncased", "classification", device = "cpu")
#' inputs <- tokenize_texts(model_info$tokenizer, c("This is a test sentence."))
#' result <- run_inference(model_info$model, inputs, "classification", list(max_length = 128))
#' print(result)
#' }
#'
#' @export
run_inference <- function(model, inputs, task, params) {
  cat("Running inference on task:", task, "\n")
  if (task == "generation") {
    result <- model$generate(
      inputs$input_ids,
      max_length  = as.integer(params$max_length),
      temperature = params$temperature,
      num_beams   = as.integer(params$num_beams)
    )
  } else if (task %in% c("classification", "token-classification", "question-answering", "summarization")) {
    result <- model(inputs)
  } else {
    stop("Unsupported task for inference.")
  }
  cat("Inference completed.\n")
  result
}

#' Process Classification Output
#'
#' Processes the output from a classification task by extracting logits, calculating softmax probabilities, 
#' and determining the predicted class for each input.
#'
#' @param outputs The raw outputs from the classification model, typically containing logits.
#'
#' @return A tibble containing the probabilities and predicted class for each input.
#'
#' @details
#' This function extracts the logits from the model output, computes the softmax probabilities for each class, 
#' and returns the predicted class for each input.
#'
#' @import tibble
#' @examples
#' \dontrun{
#' logits <- model(inputs)
#' result <- process_classification_output(logits)
#' print(result)
#' }
#' 
#' @export
process_classification_output <- function(outputs) {
  logits <- as.array(outputs$logits$detach()$cpu()$numpy())
  probabilities <- t(apply(logits, 1, softmax))
  predicted_class <- max.col(probabilities)
  probabilities_list <- split(probabilities, row(probabilities))
  tibble(
    probabilities   = probabilities_list,
    predicted_class = predicted_class
  )
}

#' Process Generation Output
#'
#' Processes the output from a text generation task by decoding the generated sequences back into text.
#'
#' @param outputs The raw output sequences from the generation model.
#' @param tokenizer The tokenizer used to decode the generated tokens.
#'
#' @return A tibble containing the generated text.
#'
#' @details
#' This function takes the raw generated sequences from the model and decodes them using the provided tokenizer 
#' to return the generated text.
#'
#' @import tibble
#' @examples
#' \dontrun{
#' generated_sequences <- model$generate(input_ids)
#' result <- process_generation_output(generated_sequences, tokenizer)
#' print(result)
#' }
#'
#' @export
process_generation_output <- function(outputs, tokenizer) {
  generated_sequences <- outputs$detach()$cpu()$numpy()
  generated_texts <- apply(generated_sequences, 1, function(tokens) {
    tokenizer$decode(as.integer(tokens), skip_special_tokens = TRUE)
  })
  tibble(generated_text = generated_texts)
}

#' Process Token Classification Output
#'
#' Processes the output from a token classification task by determining the predicted tokens for each input.
#'
#' @param outputs The raw outputs from the token classification model, typically containing logits.
#'
#' @return A tibble containing the predicted tokens for each input.
#'
#' @details
#' This function extracts the logits from the model output, applies the `which.max` function to determine 
#' the predicted token for each position in the input, and returns a tibble of the predicted tokens.
#'
#' @import tibble
#' @examples
#' \dontrun{
#' token_logits <- model(inputs)
#' result <- process_token_classification_output(token_logits)
#' print(result)
#' }
#' 
#' @export
process_token_classification_output <- function(outputs) {
  logits <- as.array(outputs$logits$detach()$cpu()$numpy())
  predicted_tokens <- apply(logits, c(1, 2), which.max)
  predicted_tokens_list <- split(predicted_tokens, seq_len(nrow(predicted_tokens)))
  tibble(predicted_tokens = predicted_tokens_list)
}

#' Process Question Answering Output
#'
#' Processes the output from a question answering task by extracting the start and end logits, then decoding 
#' the predicted answer from the input tokens.
#'
#' @param outputs The raw outputs from the question answering model, typically containing start and end logits.
#' @param tokenizer The tokenizer used to decode the tokens into text.
#' @param inputs The input token ids corresponding to the question and context.
#'
#' @return A tibble containing the predicted answer, as well as the start and end positions of the answer in the context.
#'
#' @details
#' This function extracts the start and end logits, uses them to find the span of the predicted answer, 
#' and decodes the answer from the input tokens.
#'
#' @import tibble
#' @examples
#' \dontrun{
#' qa_outputs <- model(inputs)
#' result <- process_question_answering_output(qa_outputs, tokenizer, inputs)
#' print(result)
#' }
#'
#' @export
process_question_answering_output <- function(outputs, tokenizer, inputs) {
  start_logits <- as.array(outputs$start_logits$detach()$cpu()$numpy())
  end_logits   <- as.array(outputs$end_logits$detach()$cpu()$numpy())
  predicted_start <- apply(start_logits, 1, which.max)
  predicted_end   <- apply(end_logits, 1, which.max)
  
  input_ids <- as.array(inputs$input_ids$detach()$cpu()$numpy())
  answers <- mapply(function(start, end, ids) {
    if (start > end) {
      return("")
    }
    answer_tokens <- ids[start:end]
    tokenizer$decode(as.integer(answer_tokens), skip_special_tokens = TRUE)
  }, predicted_start, predicted_end, split(input_ids, seq_len(nrow(input_ids))))
  
  tibble(
    answer = answers,
    start  = predicted_start,
    end    = predicted_end
  )
}

#' Process Model Outputs
#'
#' Processes model outputs based on the task type. The output processing includes tasks like classification, 
#' generation, token classification, and question answering.
#'
#' @param outputs The raw outputs from the model.
#' @param tokenizer The tokenizer used to decode tokens into text for generation tasks.
#' @param task Character string specifying the task (e.g., "classification", "generation").
#' @param inputs Optional. The input tokens for tasks like question answering.
#'
#' @return A processed output, typically a tibble.
#'
#' @details
#' This function processes outputs from different NLP tasks, automatically selecting the appropriate 
#' processing method based on the task.
#'
#' @import tibble
#' @examples
#' \dontrun{
#' outputs <- model(inputs)
#' result <- process_outputs(outputs, tokenizer, "classification")
#' print(result)
#' }
#'
#' @export
process_outputs <- function(outputs, tokenizer, task, inputs = NULL) {
  switch(
    task,
    "classification"       = process_classification_output(outputs),
    "generation"           = process_generation_output(outputs, tokenizer),
    "token-classification" = process_token_classification_output(outputs),
    "question-answering"   = process_question_answering_output(outputs, tokenizer, inputs),
    stop("Unsupported task for output processing.")
  )
}

#' Process a Batch of Inputs
#'
#' Processes a batch of inputs by tokenizing them, running inference, and then processing the outputs.
#'
#' @param model The pre-trained model to use for inference.
#' @param tokenizer The tokenizer used to prepare the inputs for the model.
#' @param batch_texts A character vector containing the texts to process in the batch.
#' @param task A character string specifying the task (e.g., "classification", "generation").
#' @param params A list of additional parameters for processing the batch (e.g., max_length).
#'
#' @return A tibble containing the processed outputs for the batch.
#'
#' @details
#' This function handles the tokenization of a batch of inputs, runs model inference, and processes the 
#' model outputs based on the task type.
#'
#' @import tibble
#' @examples
#' \dontrun{
#' batch_texts <- c("This is sentence one.", "This is sentence two.")
#' result <- process_batch(model, tokenizer, batch_texts, "classification", list(max_length = 128))
#' print(result)
#' }
#'
#' @export
process_batch <- function(model, tokenizer, batch_texts, task, params) {
  inputs  <- tokenize_texts(tokenizer, batch_texts, max_length = params$max_length)
  outputs <- run_inference(model, inputs, task, params)
  process_outputs(outputs, tokenizer, task, inputs)
}

#' Batch Inference with Progress Bar
#'
#' Processes multiple batches of inputs with a progress bar, tokenizes the inputs, runs model inference, 
#' and processes the outputs.
#'
#' @param model The pre-trained model to use for inference.
#' @param tokenizer The tokenizer used to prepare the inputs for the model.
#' @param texts A character vector of texts to process.
#' @param task A character string specifying the task (e.g., "classification", "generation").
#' @param params A list of additional parameters, such as batch size and max length.
#'
#' @return A tibble containing the processed outputs for all batches.
#'
#' @details
#' This function splits the input texts into batches, processes each batch, and displays a progress bar 
#' to indicate progress.
#'
#' @import progress
#' @import tibble
#' @examples
#' \dontrun{
#' texts <- c("This is a test sentence.", "Here is another sentence.")
#' result <- infer_batches(model, tokenizer, texts, "classification", list(batch_size = 2, max_length = 128))
#' print(result)
#' }
#'
#' @export
infer_batches <- function(model, tokenizer, texts, task, params) {
  batches <- split(texts, ceiling(seq_along(texts) / params$batch_size))
  pb <- progress_bar$new(
    total = length(batches),
    format = "Processing [:bar] :percent"
  )
  
  results <- lapply(batches, function(batch_texts) {
    result <- process_batch(model, tokenizer, batch_texts, task, params)
    pb$tick()
    result
  })
  
  do.call(rbind, results)
}

#' Run Model Workflow
#'
#' Runs the full model workflow for a specified NLP task, including loading the model, preprocessing the input texts, 
#' and processing the outputs.
#'
#' @param model_name Character string specifying the name of the model to load from the Hugging Face model hub.
#' @param task Character string specifying the task (e.g., "classification", "generation").
#' @param texts A character vector of texts to process.
#' @param params A list of additional parameters, such as device, batch size, and max length.
#' @param preprocess_text Logical indicating whether to preprocess the texts. Default is `TRUE`.
#' @param batch Logical indicating whether to process the texts in batches. Default is `TRUE`.
#'
#' @return A tibble containing the processed outputs.
#'
#' @details
#' This function provides an end-to-end workflow for running NLP tasks using Hugging Face models in R. It supports 
#' preprocessing, batching, and task-specific output processing.
#'
#' @import tibble
#' @examples
#' \dontrun{
#' texts <- c("This is an example sentence.")
#' params <- list(device = "cpu", batch_size = 1, max_length = 128)
#' result <- run_model_workflow("distilbert-base-uncased", "classification", texts, params)
#' print(result)
#' }
#'
#' @export
run_model_workflow <- function(model_name, task, texts, params, preprocess_text = TRUE, batch = TRUE) {
  model_info <- load_model_tokenizer(model_name, task, params$device)
  model     <- model_info$model
  tokenizer <- model_info$tokenizer
  
  if (preprocess_text) {
    texts <- preprocess_texts(texts)
  }
  
  if (batch) {
    infer_batches(model, tokenizer, texts, task, params)
  } else {
    result <- process_batch(model, tokenizer, texts, task, params)
    result
  }
}

################################################################################ API

#' Hugging Face Model Inference API Request
#'
#' Sends input data to a specified Hugging Face model for inference using the Hugging Face Inference API. The function supports retries and exponential backoff in case of failures.
#'
#' @param model_name A character string representing the name of the Hugging Face model to use for inference.
#' @param input_data A character string or a list of strings containing the input data to send to the model. Must be in a format suitable for the model.
#' @param retries An integer specifying the number of retry attempts in case of a failed request. Default is `3`.
#' @param pause A numeric value indicating the initial pause duration (in seconds) between retries. Default is `5`.
#' @param api_key A character string representing the Hugging Face API key. If `NULL`, the API key will be retrieved using the `get_api_key` function.
#' @param verbose A logical flag indicating whether to display detailed messages about the request process. Default is `TRUE`.
#'
#' @details
#' This function constructs an API request to send the input data to the Hugging Face model specified by `model_name`. The `POST` request is sent to the Hugging Face Inference API. If the request fails or a rate limit is hit, the function retries the request with exponential backoff, up to the specified number of retries.
#'
#' The input data must be either a character string or a list of strings, as expected by the model.
#'
#' @return The parsed response from the Hugging Face Inference API. If all retry attempts fail, the function throws an error.
#'
#' @examples
#' \dontrun{
#' # Perform inference with a Hugging Face model
#' response <- hf_api_inference(
#'   model_name = "distilbert-base-uncased",
#'   input_data = "This is a test input.",
#'   api_key = "your_api_key_here"
#' )
#' 
#' # Perform inference with retries and a list of inputs
#' response <- hf_api_inference(
#'   model_name = "gpt2",
#'   input_data = list("Input 1", "Input 2"),
#'   retries = 5,
#'   pause = 10,
#'   api_key = "your_api_key_here"
#' )
#' }
#'
#' @export
hf_api_inference <- function(model_name, input_data, retries = 3, pause = 5, api_key = NULL, verbose = TRUE) {
  url <- sprintf("https://api-inference.huggingface.co/models/%s", model_name)
  
  # Prepare the input data
  if (is.character(input_data) || is.list(input_data)) {
    body <- list(inputs = input_data)
  } else {
    stop("input_data must be a character string or a list of strings.")
  }
  
  make_api_request(
    method = "POST",
    url = url,
    body = body,
    retries = retries,
    pause = pause,
    api_key = api_key,
    verbose = verbose
  )
}

#' Search Hugging Face Models by Task
#'
#' Searches for Hugging Face models based on a specified task using the Hugging Face API. The function supports retries and exponential backoff in case of failed requests.
#'
#' @param task A character string representing the task for which models should be filtered (e.g., "text-classification", "translation").
#' @param retries An integer specifying the number of retry attempts in case of a failed request. Default is `3`.
#' @param pause A numeric value indicating the initial pause duration (in seconds) between retries. Default is `5`.
#' @param api_key A character string representing the Hugging Face API key. If `NULL`, the API key will be retrieved using the `get_api_key` function.
#' @param verbose A logical flag indicating whether detailed messages should be displayed during the request process. Default is `TRUE`.
#'
#' @details
#' This function constructs a `GET` request to the Hugging Face API to search for models that support the specified task. 
#' If the request fails or hits a rate limit, the function retries the request using exponential backoff, up to the specified number of retries.
#'
#' The task filter allows users to search for models based on their specific use case, such as tasks like "text-generation", "sentiment-analysis", or "summarization".
#'
#' @return A parsed response from the Hugging Face API containing model information related to the specified task. If all retries fail, the function stops with an error.
#'
#' @examples
#' \dontrun{
#' # Search for models related to text classification
#' models <- hf_api_model_search(
#'   task = "text-classification",
#'   api_key = "your_api_key_here"
#' )
#' 
#' # Search for models with a different task and more retries
#' models <- hf_api_model_search(
#'   task = "translation",
#'   retries = 5,
#'   pause = 10,
#'   api_key = "your_api_key_here"
#' )
#' }
#'
#' @export
hf_api_model_search <- function(task, retries = 3, pause = 5, api_key = NULL, verbose = TRUE) {
  url <- sprintf("https://huggingface.co/api/models?filter=task:%s", task)
  
  make_api_request(
    method = "GET",
    url = url,
    retries = retries,
    pause = pause,
    api_key = api_key,
    verbose = verbose
  )
}

#' Batch Inference with Hugging Face Model
#'
#' Performs inference on a list of texts in batches using the specified Hugging Face model via the Hugging Face Inference API. The function processes the texts in smaller batches to handle large datasets efficiently, with support for retries and exponential backoff in case of failures.
#'
#' @param model_name A character string representing the name of the Hugging Face model to use for inference.
#' @param texts A character vector or list of strings containing the input texts to be processed in batches.
#' @param batch_size An integer specifying the number of texts to include in each batch. Default is `8`.
#' @param retries An integer specifying the number of retry attempts in case of a failed request. Default is `3`.
#' @param pause A numeric value indicating the initial pause duration (in seconds) between retries. Default is `5`.
#' @param api_key A character string representing the Hugging Face API key. If `NULL`, the API key will be retrieved using the `get_api_key` function.
#' @param verbose A logical flag indicating whether detailed messages should be displayed during the batch inference process. Default is `TRUE`.
#'
#' @details
#' This function splits a list of input texts into smaller batches, each containing up to `batch_size` texts. It performs inference on each batch using the Hugging Face model specified by `model_name`. After processing all batches, it returns the combined results.
#'
#' The function supports retries with exponential backoff to handle errors or rate limits that may occur during API requests. 
#' It is particularly useful when performing inference on a large number of texts.
#'
#' @return A list containing the results of the inference for all input texts.
#'
#' @examples
#' \dontrun{
#' # Perform batch inference with a Hugging Face model
#' texts <- c("This is the first input.", "This is the second input.", "And so on...")
#' results <- hf_api_inference_batch(
#'   model_name = "distilbert-base-uncased",
#'   texts = texts,
#'   batch_size = 4,
#'   api_key = "your_api_key_here"
#' )
#' 
#' # Perform batch inference with different batch size and retries
#' results <- hf_api_inference_batch(
#'   model_name = "gpt2",
#'   texts = texts,
#'   batch_size = 10,
#'   retries = 5,
#'   api_key = "your_api_key_here"
#' )
#' }
#'
#' @export
hf_api_inference_batch <- function(model_name, texts, batch_size = 8, retries = 3, pause = 5, api_key = NULL, verbose = TRUE) {
  results <- list()
  batches <- split(texts, ceiling(seq_along(texts) / batch_size))
  
  for (batch_index in seq_along(batches)) {
    if (verbose) message(sprintf("Processing batch %d/%d", batch_index, length(batches)))
    batch <- batches[[batch_index]]
    output <- hf_api_inference(
      model_name,
      batch,
      retries = retries,
      pause = pause,
      api_key = api_key,
      verbose = verbose
    )
    results <- c(results, output)
  }
  return(results)
}

#' Compare Local Model Inference vs Hugging Face API Inference
#'
#' Compares the inference performance and results between a local model and a Hugging Face model accessed via the Hugging Face Inference API. The function measures execution time and outputs for both the local and API models.
#'
#' @param local_model The locally loaded model to use for inference. This model should be compatible with the `run_inference` function.
#' @param tokenizer The tokenizer function or object to preprocess input texts for the local model.
#' @param api_model_name A character string representing the name of the Hugging Face model to use for API inference.
#' @param texts A character vector containing the input texts for inference.
#' @param retries An integer specifying the number of retry attempts in case of a failed API request. Default is `3`.
#' @param pause A numeric value indicating the initial pause duration (in seconds) between retries for the API request. Default is `5`.
#' @param api_key A character string representing the Hugging Face API key. If `NULL`, the API key will be retrieved using the `get_api_key` function.
#' @param verbose A logical flag indicating whether detailed messages should be displayed during the process. Default is `TRUE`.
#'
#' @details
#' This function compares the inference times and outputs of a local model with those of a Hugging Face model accessed via the API. The user must provide the `tokenize_texts` and `run_inference` functions for the local model processing. It returns the inference times and results for both the local and API-based models.
#'
#' The outputs from both the local and API models are processed using the `process_local_output` and `process_api_output` functions (which should be defined by the user), allowing for a standardized comparison.
#'
#' @return A list containing two elements:
#' \describe{
#'   \item{local}{A list with the local model's inference time (in seconds) and outputs.}
#'   \item{api}{A list with the API model's inference time (in seconds) and outputs.}
#' }
#'
#' @examples
#' \dontrun{
#' # Compare a local model with a Hugging Face model for inference
#' comparison <- compare_local_vs_api(
#'   local_model = my_local_model,
#'   tokenizer = my_tokenizer,
#'   api_model_name = "distilbert-base-uncased",
#'   texts = c("This is a test.", "Comparing models."),
#'   api_key = "your_api_key_here"
#' )
#'
#' # View the comparison of times and outputs
#' print(comparison)
#' }
#'
#' @export
compare_local_vs_api <- function(
    local_model,
    tokenizer,
    api_model_name,
    texts,
    retries = 3,
    pause = 5,
    api_key = NULL,
    verbose = TRUE
) {
  if (!exists("tokenize_texts") || !exists("run_inference")) {
    stop("Functions 'tokenize_texts' and 'run_inference' must be defined for local model inference.")
  }
  
  # Placeholder functions for processing outputs (user should define these)
  process_local_output <- function(output) {
    # Convert local model output to a standardized format
    return(output)
  }
  
  process_api_output <- function(output) {
    # Convert API model output to a standardized format
    return(output)
  }
  
  # Local model inference
  local_time <- system.time({
    local_outputs <- lapply(texts, function(text) {
      inputs <- tokenize_texts(tokenizer, text)
      output <- run_inference(local_model, inputs)
      process_local_output(output)
    })
  })
  
  # API model inference
  api_time <- system.time({
    api_outputs <- lapply(texts, function(text) {
      output <- hf_api_inference(
        api_model_name,
        text,
        retries = retries,
        pause = pause,
        api_key = api_key,
        verbose = verbose
      )
      process_api_output(output)
    })
  })
  
  # Comparison summary
  list(
    local = list(time = as.numeric(local_time["elapsed"]), outputs = local_outputs),
    api = list(time = as.numeric(api_time["elapsed"]), outputs = api_outputs)
  )
}

#' Benchmark Model Performance
#'
#' Benchmarks the performance of a model by measuring the total inference time and calculating accuracy (if expected labels are provided). The function applies a model to a list of texts and supports retries with exponential backoff.
#'
#' @param model_function A function representing the model inference method. The function should accept a text input and return an inference result.
#' @param texts A character vector containing the input texts for benchmarking.
#' @param retries An integer specifying the number of retry attempts in case of a failed request. Default is `3`.
#' @param pause A numeric value indicating the initial pause duration (in seconds) between retries. Default is `5`.
#' @param api_key A character string representing the API key to be passed to the model function, if needed. Default is `NULL`.
#' @param expected_labels A list of expected labels for comparison with model results. If provided, accuracy will be computed. Default is `NULL`.
#'
#' @details
#' This function benchmarks the performance of a model by applying `model_function` to a list of input texts. It measures the total time taken to perform inference and, if `expected_labels` are provided, calculates the accuracy of the model's predictions.
#'
#' The function supports retries for handling failed API requests or other issues, with an exponential backoff mechanism. It is suitable for evaluating model performance both in terms of time and accuracy.
#'
#' @return A list containing:
#' \describe{
#'   \item{total_time}{The total time (in seconds) taken to run inference on all texts.}
#'   \item{accuracy}{The accuracy of the model's predictions compared to `expected_labels`, if provided. If `expected_labels` is `NULL`, accuracy is set to `NA`.}
#'   \item{results}{A list of model inference results for each text.}
#' }
#'
#' @examples
#' \dontrun{
#' # Benchmark a model with provided texts
#' benchmark <- benchmark_model_performance(
#'   model_function = hf_api_inference,
#'   texts = c("This is a test.", "Benchmarking model."),
#'   api_key = "your_api_key_here"
#' )
#'
#' # Benchmark with expected labels for accuracy
#' benchmark <- benchmark_model_performance(
#'   model_function = hf_api_inference,
#'   texts = c("This is a test.", "Benchmarking model."),
#'   expected_labels = list("label1", "label2"),
#'   api_key = "your_api_key_here"
#' )
#' 
#' # View the benchmark results
#' print(benchmark)
#' }
#'
#' @export
benchmark_model_performance <- function(
    model_function,
    texts,
    retries = 3,
    pause = 5,
    api_key = NULL,
    expected_labels = NULL
) {
  start_time <- Sys.time()
  
  results <- lapply(texts, function(text) {
    model_function(text, retries = retries, pause = pause, api_key = api_key)
  })
  
  total_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  
  accuracy <- NA
  if (!is.null(expected_labels)) {
    if (length(results) != length(expected_labels)) {
      stop("Length of results and expected labels do not match.")
    }
    correct <- sum(mapply(function(res, label) {
      identical(res, label)
    }, results, expected_labels))
    accuracy <- correct / length(expected_labels)
  }
  
  list(total_time = total_time, accuracy = accuracy, results = results)
}