### Final Code:
Here is the final code used for this analysis:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
file_path = '/mnt/data/customer_churn.csv'
customer_churn_df = pd.read_csv(file_path)
# Display the first few rows of the dataset
customer_churn_df.head()
# Initial data analysis
print(customer_churn_df.info())
print(customer_churn_df.describe())
print(customer_churn_df['Churn'].value_counts())

# Data cleaning and preprocessing
customer_churn_df['Onboard_date'] = pd.to_datetime(customer_churn_df['Onboard_date'])
customer_churn_df['Years_since_onboard'] = (pd.to_datetime('2023-01-01') - customer_churn_df['Onboard_date']).dt.days / 365.25
customer_churn_df.drop(columns=['Names', 'Onboard_date', 'Location', 'Company'], inplace=True)

# Splitting the data
X = customer_churn_df.drop(columns=['Churn'])
y = customer_churn_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = ['Age', 'Total_Purchase', 'Years', 'Num_Sites', 'Years_since_onboard']
numeric_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='mean')),
  ('scaler', StandardScaler())
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

# Model training pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42))])

model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(classification_rep)
print(conf_matrix)
