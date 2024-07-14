# Auto-Kaggle Project

The Auto-Kaggle project aims to develop a principled pipeline that utilizes Language Models (LLMs) such as ChatGPT to automatically solve Kaggle-like tasks on structured data and achieve competitive scores. This involves creating a pipeline that processes structured data, generates predictions for target variables, and handles various data formats while optimizing performance and accuracy.

## Project Overview

### High-Level Description
The research project focuses on using LLMs to perform data scientist tasks autonomously. The pipeline:
- Understands the data structure and target variables.
- Generates code for data analysis and modifications.
- Produces code to train models and generate predictions.

### Complications Addressed
- Controlling the LLM to ensure reliability.
- Handling code errors and minimizing API calls.
- Ensuring minimal human intervention and high adaptability.
- Proper evaluation to avoid overfitting to specific datasets.

## Code Structure

auto_kaggler_proj/
│
├── automl_models_for_comparison/
│
├── config/
│
├── data/
│
├── environment/
│   ├── evaluation_logs/
│   └── validation_logs/
│       ├── input/
│       │   └── generated_code_<task_type>.py
│       └── output/
│           └── validation_log_<task_type>.py
│
├── generated_scripts/
│   ├── anomaly_detection/
│   ├── classification/
│   ├── forecasting/
│   ├── generated_code_examples/
│   └── regression/
│
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── auto_fix_script.py
│   ├── code_execution.py
│   ├── code_validator.py
│   ├── config_loader.py
│   ├── list_models.py
│   ├── llm_base.py
│   ├── llm_gpt4.py
│   ├── llm_gpt35.py
│   ├── logging_config.py
│   ├── pipeline_manager.py
├── __init__.py
|── diagram.png
|── main.py
|── README.md
└── requirements.txt

## Key Components

### Pipeline Manager (`pipeline_manager.py`)
- Sends prompts to the LLM to generate code for each step.
- Executes the generated code and handles execution errors by prompting the LLM for corrections.
- Saves the generated code to a file (`generated_code.py`) for validation.

### Code Validator (`code_validator.py`)
- Performs validation checks on the generated code, including syntax, PEP 8 compliance, API calls, comments, and function docstrings.
- Logs results of each check to a log file (`validation_log.txt`).
- Provides a total score based on the number of checks passed.

### Workflow
1. The pipeline manager generates and executes code for each step.
2. Generated code is saved to a file (`generated_code.py`).
3. The validation script validates the generated code.
4. If validation fails, the pipeline manager prompts the LLM to correct the code and revalidates it.
5. This process iterates until the code passes validation.

### Integration Steps
1. **Generate Code**: The pipeline manager generates code snippets for each step.
2. **Validate Code**: The code is validated using the validation module.
3. **Request Fixes**: If validation fails, request the LLM to fix the issues.
4. **Revalidate**: Revalidate the corrected code.

### Optimization Steps
1. **Data Preprocessing**:
   - Handle various data formats (CSV, JSON, Excel, etc.)
   - Impute missing values
   - Normalize/Standardize numerical features
   - Encode categorical features
   - Feature engineering and selection
2. **Model Selection and Training**:
   - Experiment with different algorithms (e.g., RandomForest, XGBoost, SVM, etc.)
   - Use cross-validation to assess model performance
   - Implement an automatic hyperparameter tuning mechanism (e.g., Grid Search, Random Search, Bayesian Optimization)
3. **Improve Performance (compared with benchmark results)**:
   - Perform hyperparameter tuning to find the best parameters for the chosen model
   - LLM model asked to improve performance using a combination of these methods which can significantly enhance model performance and robustness.

   3.1 **Feature Engineering**:
      - **Create Interaction Features**: Combine features to capture interactions.
      - **Polynomial Features**: Add polynomial terms to capture non-linear relationships.
      - **Domain-Specific Features**: Create features based on domain knowledge.

   3.2 **Handling Class Imbalance**:
      - **Oversampling**: Use techniques like SMOTE to balance the classes.
      - **Undersampling**: Reduce the number of samples in the majority class.
      - **Class Weights**: Adjust the weights of the classes in the loss function.

   3.3 **Hyperparameter Tuning**:
      - **Grid Search**: Exhaustively search over a specified parameter grid.
      - **Random Search**: Randomly sample parameters from a specified distribution.
      - **Bayesian Optimization**: Use probabilistic models to find the optimal hyperparameters.

   3.4 **Ensemble Methods**:
      - **Bagging**: Combine multiple models trained on different subsets of the data (e.g., Random Forest).
      - **Boosting**: Sequentially train models to correct the errors of previous models (e.g., XGBoost, AdaBoost).
      - **Stacking**: Combine predictions from multiple models using a meta-model.

   3.5 **Model Selection**:
      - **Try Different Algorithms**: Experiment with different machine learning algorithms to find the best one for your data.
      - **Model Complexity**: Adjust the complexity of the model (e.g., depth of decision trees, number of layers in neural networks).

   3.6 **Data Preprocessing**:
      - **Scaling/Normalization**: Scale numerical features to have a similar range.
      - **Imputation**: Handle missing values using strategies like mean, median, or mode imputation.
      - **Encoding**: Convert categorical features into numerical format (e.g., One-Hot Encoding, Label Encoding).

   3.7 **Cross-Validation**:
      - **K-Fold Cross-Validation**: Divide the data into k subsets and train/test the model k times.
      - **Stratified Sampling**: Ensure that each fold has a similar distribution of classes.

   3.8 **Regularization**:
      - **L1/Lasso**: Adds a penalty for non-zero coefficients, encouraging sparsity.
      - **L2/Ridge**: Adds a penalty for large coefficients, encouraging smaller coefficients.

   3.9 **Early Stopping**:
      - **Monitor Validation Performance**: Stop training when the performance on a validation set stops improving to prevent overfitting.

   3.10 **Data Augmentation**:
      - **Synthetic Data**: Generate synthetic data to increase the size and diversity of the training set.
      - **Transformations**: Apply transformations (e.g., rotations, scaling) to create new training samples.

4. **Model Evaluation**:
   - Evaluate models using appropriate metrics (accuracy, precision, recall, F1 score, ROC-AUC, etc.)
5. **Iterative Improvement**:
   - Based on evaluation, iteratively improve preprocessing steps, feature engineering, model selection, and hyperparameter tuning

6. **Comparison with Benchmarks**:
   - Compare the model's performance with the benchmark solution found at the specified URL in the `config.yaml` file.
   - Additionally, the LLM is instructed to find other benchmarks from Kaggle or other communities for a comprehensive performance evaluation.

## How to Run

1. **Set Up the Environment**:
   - Ensure you have Python installed.
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Set the OpenAI API Key**:
   - Export the OpenAI API key as an environment variable:
     ```bash
     export OPENAI_API_KEY=your_openai_api_key
     ```

3. **Run the Pipeline**:
   - Execute the main script with the necessary arguments:
     ```bash
     python main.py --config path_to_config_file --model gpt4 --dataset dataset_name
     ```

### Main Script (`main.py`)
The main script sets up the argument parser, loads the configuration, initializes the LLM, and runs the pipeline.

## Diagram

[Diagram](diagram.pdf)

## Contact
For any inquiries or issues, please feel free to contact me.
