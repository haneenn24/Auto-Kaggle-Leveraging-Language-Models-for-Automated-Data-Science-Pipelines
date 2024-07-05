import openai
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from code_validator import validate_script, log_message

# Initialization
openai.api_key = os.getenv("OPENAI_API_KEY")

def send_prompt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1500,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Function to execute generated code
def execute_code(code, globals=None, locals=None):
    try:
        exec(code, globals, locals)
        return None
    except Exception as e:
        return str(e)

# Function to handle code generation, validation, and correction
def generate_and_validate(prompt):
    code = send_prompt(prompt)
    print("Generated Code:\n", code)
    error = execute_code(code, globals(), locals())
    
    if error:
        log_message("validation_log.txt", "Execution Error", "Code execution failed.", error)
        debug_prompt = f"""
        There was an error in the following code:
        {code}
        The error message was: {error}
        Please correct the code and provide a new version.
        """
        code = send_prompt(debug_prompt)
        print("Corrected Code:\n", code)
        error = execute_code(code, globals(), locals())
    
    with open("generated_code.py", "w") as file:
        file.write(code)
    
    validate_script("generated_code.py")

# Step 1: Load Data
file_path = './data/customer_churn.csv'
data_prompt = f"""
You are an AI data scientist tasked with solving a customer churn prediction problem. 
Load the data from the following path and provide basic analysis:
{file_path}
"""
generate_and_validate(data_prompt)

# Step 2: Data Analysis and Preprocessing
analysis_prompt = f"""
Analyze the dataset and preprocess it for machine learning. 
Handle missing values, convert categorical variables, and create any necessary features.
"""
generate_and_validate(analysis_prompt)

# Step 3: Model Selection and Hyperparameter Tuning
model_prompt = f"""
Train a machine learning model to predict the Churn variable. 
Use cross-validation and perform hyperparameter tuning to optimize model performance. 
Evaluate its performance using appropriate metrics.
"""
generate_and_validate(model_prompt)

# Step 4: Performance Optimization 
benchmark_prompt = f"""
Compare the model's performance to benchmarks and suggest improvements.
"""
generate_and_validate(benchmark_prompt)

# Step 5: Iterative Improvement
improvement_prompt = f"""
Iteratively improve the model's performance based on the comparison with benchmarks.
"""
generate_and_validate(improvement_prompt)
