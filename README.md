# face_huggeR

**face_huggeR** is a comprehensive R package designed to streamline the use of Hugging Face models within R through `reticulate`. It provides an end-to-end solution for Natural Language Processing (NLP) tasks such as text classification, generation, token classification, and question answering. This package seamlessly integrates model loading, tokenization, text preprocessing, inference execution, and output processing, making it easier for R users to work with powerful NLP models from Hugging Face.

### Key Features:
- **Model Loading & Tokenization**: Load pre-trained models and tokenizers from Hugging Face’s model hub directly into R for a variety of NLP tasks.
- **Inference Support**: Run model inference for tasks like text classification, generation, and question answering with minimal setup.
- **Batch Processing**: Efficiently handle large datasets by splitting input texts into batches and providing a progress bar for batch inference.
- **NLP Preprocessing**: Includes built-in functions for text preprocessing such as lowercasing and stopword removal, as well as tokenization and processing of model outputs.
- **Device Management**: Automatically detects CUDA availability for GPU-based inference, improving model performance when possible.

The `face_huggeR.R` script contains essential functions for model interactions, while `utilities.R` provides utility functions to assist in validating tasks, devices, and handling common NLP preprocessing tasks like softmax calculation and stopword removal.

This package enables R users to leverage the power of Hugging Face models for NLP tasks with ease, providing flexibility, efficiency, and ease of integration with their R workflows.


## face_huggeR.R Functions

The following script has base functions designed to interactive with models on Hugging Face. Below is a brief description of each.

### 1. `load_modules()`

**Description:**  
Loads the required Python modules (`transformers` and `torch`) for using Hugging Face models and PyTorch in R. This function leverages `reticulate` to interface between R and Python, utilizing delayed loading for efficiency. If any module fails to load, an informative error message is provided and the function stops execution.

**Usage:**
```r
modules <- load_modules()
```

### 2. `load_model_tokenizer()`

**Description:**  
Loads a Hugging Face model and tokenizer for a specified task into the desired device (CPU or CUDA). This function supports several tasks such as classification, generation, and summarization, and allows for caching models locally for faster future access. It checks for CUDA availability and automatically moves the model to CUDA if specified.

**Parameters:**
- `model_name`: A character string specifying the model name to load from the Hugging Face model hub.
- `task`: A character string specifying the type of task (e.g., "classification", "generation").
- `device`: A character string indicating the device to run the model on. Options are `"cpu"` or `"cuda"`. The default is `"cpu"`.
- `cache_dir`: An optional character string specifying the directory to cache the model. If not provided, a default directory is used.

**Usage:**
```r
model_info <- load_model_tokenizer("distilbert-base-uncased", "classification", device = "cpu")
```

### 3. `check_cuda_status()`

**Description:**  
Checks whether CUDA is available on the system and, if available, returns the number of GPUs that can be used. This function is useful for determining whether a model can be run on a GPU.

**Returns:**  
A logical value indicating whether CUDA is available (`TRUE`) or not (`FALSE`).

**Usage:**
```r
cuda_available <- check_cuda_status()
```

### 4. `preprocess_texts()`

**Description:**  
Preprocesses a character vector of texts by optionally converting them to lowercase and removing stopwords. This function is useful for preparing texts for NLP tasks.

**Parameters:**
- `texts`: A character vector of texts to preprocess.
- `lowercase`: A logical value indicating whether to convert texts to lowercase. Default is `TRUE`.
- `remove_stopwords_flag`: A logical value indicating whether to remove stopwords. Default is `FALSE`.

**Returns:**  
A character vector of preprocessed texts.

**Usage:**
```r
preprocessed_texts <- preprocess_texts(c("This is an example sentence."), lowercase = TRUE, remove_stopwords_flag = TRUE)
```

### 5. `tokenize_texts()`

**Description:**  
Tokenizes a character vector of texts using a specified tokenizer from the `transformers` library and returns input tensors for use with NLP models. This function handles padding, truncation, and converts the text into token IDs and attention masks.

**Parameters:**
- `tokenizer`: A tokenizer object from the `transformers` library.
- `texts`: A character vector of texts to tokenize.
- `max_length`: An integer specifying the maximum length of the tokenized sequences. Default is `128`.
- `return_tensors`: A character string specifying the format of the returned tensors (`"pt"` for PyTorch tensors). Default is `"pt"`.
- `padding`: A logical value indicating whether to pad the sequences to the maximum length. Default is `TRUE`.
- `truncation`: A logical value indicating whether to truncate the sequences to the maximum length. Default is `TRUE`.

**Returns:**  
A list containing `input_ids` and `attention_mask` arrays, which are the tokenized inputs ready for the model.

**Usage:**
```r
tokenizer <- load_modules()$transformers$AutoTokenizer$from_pretrained("distilbert-base-uncased")
tokenized_texts <- tokenize_texts(tokenizer, c("Hello, world!"))
```

### 6. `run_inference()`

**Description:**  
Runs inference on a specified NLP task (e.g., text generation or classification) using a pre-trained Hugging Face model. This function can handle various tasks and processes the input accordingly to return the model's output.

**Parameters:**
- `model`: The Hugging Face model object to run inference with.
- `inputs`: A list of tokenized inputs for the model (e.g., `input_ids`, `attention_mask`).
- `task`: A character string specifying the task type (e.g., "generation", "classification").
- `params`: A list of additional parameters for inference, such as `max_length`, `temperature`, and `num_beams`.

**Returns:**  
The model output after running inference, depending on the task.

**Usage:**
```r
model_info <- load_model_tokenizer("distilbert-base-uncased", "classification", device = "cpu")
inputs <- tokenize_texts(model_info$tokenizer, c("This is a test sentence."))
result <- run_inference(model_info$model, inputs, "classification", list(max_length = 128))
```

### 7. `process_classification_output()`

**Description:**  
Processes the output from a classification task by extracting logits, calculating softmax probabilities, and determining the predicted class for each input. This is typically used after running inference on classification tasks.

**Parameters:**
- `outputs`: The raw outputs from the classification model, typically containing logits.

**Returns:**  
A `tibble` containing softmax probabilities for each class and the predicted class for each input.

**Usage:**
```r
logits <- model(inputs)
result <- process_classification_output(logits)
```

### 8. `process_generation_output()`

**Description:**  
Processes the output from a text generation task by decoding the generated sequences into human-readable text using a specified tokenizer. This function is typically used after generating text with a model.

**Parameters:**
- `outputs`: The raw output sequences from the generation model.
- `tokenizer`: The tokenizer used to decode the generated tokens.

**Returns:**  
A `tibble` containing the generated text for each sequence.

**Usage:**
```r
generated_sequences <- model$generate(input_ids)
result <- process_generation_output(generated_sequences, tokenizer)
```

### 9. `process_token_classification_output()`

**Description:**  
Processes the output from a token classification task by extracting the predicted token for each position in the input sequence based on the model's logits. It returns the predicted tokens for each input in a structured format.

**Parameters:**
- `outputs`: The raw outputs from the token classification model, typically containing logits.

**Returns:**  
A `tibble` containing the predicted tokens for each input sequence.

**Usage:**
```r
token_logits <- model(inputs)
result <- process_token_classification_output(token_logits)
```

### 10. `process_question_answering_output()`

**Description:**  
Processes the output from a question answering task by extracting the start and end logits, determining the span of the predicted answer, and decoding it from the input tokens.

**Parameters:**
- `outputs`: The raw outputs from the question answering model, typically containing start and end logits.
- `tokenizer`: The tokenizer used to decode the tokens into text.
- `inputs`: The input token IDs corresponding to the question and context.

**Returns:**  
A `tibble` containing the predicted answer along with the start and end positions of the answer within the context.

**Usage:**
```r
qa_outputs <- model(inputs)
result <- process_question_answering_output(qa_outputs, tokenizer, inputs)
```

### 11. `process_outputs()`

**Description:**  
Processes the raw model outputs based on the specified NLP task. This function handles tasks such as classification, generation, token classification, and question answering, applying the appropriate output processing method for each task.

**Parameters:**
- `outputs`: The raw outputs from the model.
- `tokenizer`: The tokenizer used to decode tokens for generation tasks.
- `task`: A character string specifying the task (e.g., "classification", "generation").
- `inputs`: Optional. The input tokens for tasks like question answering.

**Returns:**  
A processed output, typically returned as a `tibble` depending on the task type.

**Usage:**
```r
outputs <- model(inputs)
result <- process_outputs(outputs, tokenizer, "classification")
```

### 12. `process_batch()`

**Description:**  
Processes a batch of inputs by tokenizing the texts, running inference with the model, and processing the outputs based on the specified task. This function is useful for handling multiple inputs in one go, streamlining the tokenization, inference, and output processing steps.

**Parameters:**
- `model`: The pre-trained model used for inference.
- `tokenizer`: The tokenizer used to prepare the inputs for the model.
- `batch_texts`: A character vector containing the texts to process in the batch.
- `task`: A character string specifying the task (e.g., "classification", "generation").
- `params`: A list of additional parameters for processing the batch (e.g., `max_length`).

**Returns:**  
A `tibble` containing the processed outputs for the entire batch.

**Usage:**
```r
batch_texts <- c("This is sentence one.", "This is sentence two.")
result <- process_batch(model, tokenizer, batch_texts, "classification", list(max_length = 128))
```

### 13. `infer_batches()`

**Description:**  
Processes multiple batches of inputs with a progress bar, tokenizing the texts, running model inference, and processing the outputs for each batch. This function is useful for efficiently handling large datasets by splitting the inputs into manageable batches.

**Parameters:**
- `model`: The pre-trained model used for inference.
- `tokenizer`: The tokenizer used to prepare the inputs for the model.
- `texts`: A character vector of texts to process.
- `task`: A character string specifying the task (e.g., "classification", "generation").
- `params`: A list of additional parameters, such as `batch_size` and `max_length`.

**Returns:**  
A `tibble` containing the processed outputs for all batches.

**Usage:**
```r
texts <- c("This is a test sentence.", "Here is another sentence.")
result <- infer_batches(model, tokenizer, texts, "classification", list(batch_size = 2, max_length = 128))
```

### 14. `run_model_workflow()`

**Description:**  
Runs a complete NLP model workflow for a specified task. This includes loading the model, preprocessing the input texts, running inference, and processing the outputs. It supports batch processing and text preprocessing for efficient model execution.

**Parameters:**
- `model_name`: A character string specifying the name of the model to load from the Hugging Face model hub.
- `task`: A character string specifying the task (e.g., "classification", "generation").
- `texts`: A character vector of texts to process.
- `params`: A list of additional parameters, such as `device`, `batch_size`, and `max_length`.
- `preprocess_text`: A logical value indicating whether to preprocess the texts. Default is `TRUE`.
- `batch`: A logical value indicating whether to process the texts in batches. Default is `TRUE`.

**Returns:**  
A `tibble` containing the processed outputs for the specified task.

**Usage:**
```r
texts <- c("This is an example sentence.")
params <- list(device = "cpu", batch_size = 1, max_length = 128)
result <- run_model_workflow("distilbert-base-uncased", "classification", texts, params)
```


## Utilities.R Functions

This section describes the utility functions provided in the `utilities.R` script. These functions are designed to assist in Natural Language Processing (NLP) workflows by providing task validation, device validation, softmax calculations, and stopword removal.

### 1. `validate_task()`

**Description:**  
Validates a given task against a list of supported tasks. This function ensures that only valid tasks are passed into a workflow. If an invalid task is provided, an error is thrown.

**Parameters:**
- `task`: A character string representing the task to validate (e.g., "classification", "generation").
- `valid_tasks`: A character vector containing valid task names.

**Usage:**
```r
validate_task(task, valid_tasks)
```

### 2. `validate_device()`

**Description:**  
Validates whether the provided device is either "cpu" or "cuda". If the device is not valid, the function throws an error.

**Parameters:**
- `device`: A character string specifying the device type. Can be either "cpu" or "cuda".

**Usage:**
```r
validate_device(device)
```

### 3. `softmax()`

**Description:**  
Computes the softmax of a numeric vector, transforming it into a probability distribution where all values sum to 1.

**Parameters:**
- `x`: A numeric vector for which the softmax transformation will be applied.

**Usage:**
```r
softmax(x)
```

### 4. `remove_stopwords()`

**Description:**  
Removes stopwords from a list of input texts. This function compares each word in the input text against a list of stopwords and removes any matches.

**Parameters:**
- `texts`: A character vector containing the texts from which stopwords will be removed.
- `stopwords`: A character vector of stopwords to be removed, typically from `tm::stopwords("en")`.

**Usage:**
```r
remove_stopwords(texts, stopwords)
```
