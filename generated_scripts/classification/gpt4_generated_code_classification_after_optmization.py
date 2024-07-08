import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from skopt import BayesSearchCV

# Load the dataset
file_path = 'customer_churn.csv'
customer_churn_df = pd.read_csv(file_path)

# Data cleaning and preprocessing
customer_churn_df['Onboard_date'] = pd.to_datetime(customer_churn_df['Onboard_date'])
customer_churn_df['Years_since_onboard'] = (pd.to_datetime('2023-01-01') - customer_churn_df['Onboard_date']).dt.days / 365.25
customer_churn_df.drop(columns=['Names', 'Onboard_date', 'Location', 'Company'], inplace=True)

# Splitting the data
X = customer_churn_df.drop(columns=['Churn'])
y = customer_churn_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Engineering
numeric_features = ['Age', 'Total_Purchase', 'Years', 'Num_Sites', 'Years_since_onboard']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

categorical_features = ['Account_Manager']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Define the model
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Define hyperparameters to tune using Bayesian optimization
param_grid = {
    'n_estimators': (100, 300),
    'learning_rate': (0.01, 0.2, 'log-uniform'),
    'max_depth': (3, 7),
    'subsample': (0.8, 1.0),
    'colsample_bytree': (0.8, 1.0)
}

# Set up BayesSearchCV
bayes_search = BayesSearchCV(estimator=model, search_spaces=param_grid, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('bayes_search', bayes_search)])

# Fit the model
pipeline.fit(X_train_smote, y_train_smote)

# Best parameters and model evaluation
best_params = pipeline.named_steps['bayes_search'].best_params_
best_model = pipeline.named_steps['bayes_search'].best_estimator_

# Transform the test set
X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

# Make predictions
y_pred = best_model.predict(X_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Best Parameters: {best_params}')
print(f'Accuracy: {accuracy}')
print(classification_rep)
print(conf_matrix)
