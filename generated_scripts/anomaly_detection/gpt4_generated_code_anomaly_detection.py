import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV

# Load the dataset
file_path = 'creditcard.csv'
creditcard_df = pd.read_csv(file_path)

# Initial Data Analysis
print(creditcard_df.info())
print(creditcard_df.describe())

# Assuming 'Class' is the target variable for anomaly detection
target = 'Class'

# Data cleaning and preprocessing
# Splitting the data
X = creditcard_df.drop(columns=[target])
y = creditcard_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = X.columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Define the model
model = IsolationForest(random_state=42)

# Define hyperparameters to tune using Bayesian optimization
param_grid = {
    'n_estimators': (50, 200),
    'max_samples': (0.6, 1.0),
    'contamination': (0.01, 0.1, 'log-uniform'),
    'max_features': (0.5, 1.0)
}

# Set up BayesSearchCV
bayes_search = BayesSearchCV(estimator=model, search_spaces=param_grid, n_iter=50, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)

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
y_pred = best_model.decision_function(X_test_transformed)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Best Parameters: {best_params}')
print(f'ROC AUC Score: {roc_auc}')
