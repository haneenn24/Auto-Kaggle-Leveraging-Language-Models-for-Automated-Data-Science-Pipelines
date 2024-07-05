from llm_base import LLMBase
import openai
import logging
import pandas as pd

class GPTModel4(LLMBase):
    def __init__(self, api_key):
        """
        Initialize the GPT model with the provided API key.

        Parameters:
        api_key (str): OpenAI API key.
        """
        self.api_key = api_key
        openai.api_key = api_key
        
    def generate_code(self, dataset_paths):
        """
        Generate code for the entire machine learning pipeline based on the dataset provided.

        Parameters:
        dataset_paths (tuple or str): Paths to the train and test dataset files or a single dataset file.

        Returns:
        str: Generated code as a string.
        """
        if isinstance(dataset_paths, tuple):
            train_path, test_path = dataset_paths
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            dataset_description = f"Train Dataset:\n{train_data.head().to_csv(index=False)}\nTest Dataset:\n{test_data.head().to_csv(index=False)}"
            prompt = f"""
            Generate a complete, professional Python code using PyTorch to solve a Kaggle-like task on structured data, following the steps outlined below. Ensure the code includes proper documentation, comments, and logging of main details at each stage. Please provide the full code directly without additional explanations.
            Dataset Description:
            {dataset_description}

            Tasks:

            Data Ingestion:

            Write code that reads data from a CSV file.
            Data Preprocessing:

            Handle missing values appropriately.
            Log the number of missing values in each column.
            Handling categorical variables.
            Encoding categorical variables.
            Normalize the data to ensure consistent scales.
            Split the data into training, validation, and test sets.
            Implement cross-validation by splitting the data into different sets.
            Feature and Target Identification:

            Identify the features and target variable.
            Log the identified features and target variable.
            Determine the task type (e.g., classification or regression).
            Log the determined task type.
            Model Selection:

            Choose a suitable model for the identified task type.
            Model Training:

            Train the model on all data splits created in the preprocessing step.
            Use different hyperparameters (e.g., epochs, batch size, learning rate, scheduler, optimizer) to improve model performance.
            Model Evaluation:

            Evaluate the model on all mentioned sets.
            Log the model's performance metrics, including accuracy.
            Hyperparameter Tuning:

            Perform hyperparameter tuning to achieve the best performance.
            Prediction Generation:

            Generate predictions on new data.
            Result Reporting:

            Report the results of the model's performance.
            Log the final model's performance results and accuracy.
            General Notes:

            At each stage above, write the main details to the log.
            Ensure that the code includes proper documentation and comments.
            Provide the full code directly without additional explanations.
            """
        else:
            dataset_path = dataset_paths
            data = pd.read_csv(dataset_path)
            dataset_description = f"Dataset:\n{data.head().to_csv(index=False)}"
            prompt = f"""
            Generate a complete, professional Python code using PyTorch to solve a Kaggle-like task on structured data, following the steps outlined below. Ensure the code includes proper documentation, comments, and logging of main details at each stage. Please provide the full code directly without additional explanations.
            Dataset Description:
            {dataset_description}

            Tasks:

            Data Ingestion:

            Write code that reads data from a CSV file.
            Data Preprocessing:

            Handle missing values appropriately.
            Log the number of missing values in each column.
            Handling categorical variables.
            Encoding categorical variables.
            Normalize the data to ensure consistent scales.
            Split the data into training, validation, and test sets.
            Implement cross-validation by splitting the data into different sets.
            Feature and Target Identification:

            Identify the features and target variable.
            Log the identified features and target variable.
            Determine the task type (e.g., classification or regression).
            Log the determined task type.
            Model Selection:

            Choose a suitable model for the identified task type.
            Model Training:

            Train the model on all data splits created in the preprocessing step.
            Use different hyperparameters (e.g., epochs, batch size, learning rate, scheduler, optimizer) to improve model performance.
            Model Evaluation:

            Evaluate the model on all mentioned sets.
            Log the model's performance metrics, including accuracy.
            Hyperparameter Tuning:

            Perform hyperparameter tuning to achieve the best performance.
            Prediction Generation:

            Generate predictions on new data.
            Result Reporting:

            Report the results of the model's performance.
            Log the final model's performance results and accuracy.
            General Notes:

            At each stage above, write the main details to the log.
            Ensure that the code includes proper documentation and comments.
            Provide the full code directly without additional explanations.
            """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096  # Increase max tokens to generate more comprehensive code
            )
            
            code = response['choices'][0]['message']['content'].strip()
            logging.info(f"Generated code: {code}")
            return code
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gpt_model = GPTModel(api_key="your_openai_api_key")
    dataset_paths = ("/home/sharifm/students/haneenn/auto-kaggler-proj/data/raw/customer_churn_dataset-training-master.csv", 
                     "/home/sharifm/students/haneenn/auto-kaggler-proj/data/raw/customer_churn_dataset-testing-master.csv")
    generated_code = gpt_model.generate_code(dataset_paths)
    if generated_code:
        print(generated_code)
