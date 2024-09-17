# face_huggeR

Introduction TBD.

## face_huggeR.R Functions

`face_huggeR.R` script descriptions TBD


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
