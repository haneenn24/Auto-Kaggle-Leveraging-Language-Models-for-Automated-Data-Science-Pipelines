import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from xgboost import XGBRegressor
import numpy as np

# Load the dataset
file_path = 'retail_sales.csv'  # Update the path accordingly
retail_sales_df = pd.read_csv(file_path)

# Initial Data Analysis
print(retail_sales_df.info())
print(retail_sales_df.describe())

# Assuming 'sales' is the target variable for forecasting
target = 'sales'

# Data cleaning and preprocessing
retail_sales_df.drop(columns=['Unnamed: 0'], inplace=True)  # Dropping any unnecessary columns

# Splitting the data
X = retail_sales_df.drop(columns=[target])
y = retail_sales_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Define the model
model = XGBRegressor(random_state=42, objective='reg:squarederror')

# Define hyperparameters to tune using Bayesian optimization
param_grid = {
    'n_estimators': (100, 300),
    'learning_rate': (0.01, 0.2, 'log-uniform'),
    'max_depth': (3, 7),
    'subsample': (0.8, 1.0),
    'colsample_bytree': (0.8, 1.0)
}

# Set up BayesSearchCV
bayes_search = BayesSearchCV(estimator=model, search_spaces=param_grid, n_iter=50, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42)

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('bayes_search', bayes_search)])

# Fit the model
pipeline.fit(X_train, y_train)

# Best parameters and model evaluation
best_params = pipeline.named_steps['bayes_search'].best_params_
best_model = pipeline.named_steps['bayes_search'].best_estimator_

# Transform the test set
X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

# Make predictions
y_pred = best_model.predict(X_test_transformed)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Best Parameters: {best_params}')
print(f'Root Mean Squared Error: {rmse}')
