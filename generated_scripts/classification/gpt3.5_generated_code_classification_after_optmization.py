# Load dataset (assuming customer_churn.csv is in the current directory)
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('customer_churn.csv')

# Initial data exploration
print("Dataset structure and summary statistics:")
print(df.head())
print(df.describe())

# Handle missing values (if any)
df.fillna(df.median(), inplace=True)

# Feature engineering and preprocessing
# Example: Encoding categorical variables
df = pd.get_dummies(df, columns=['Gender', 'Education', 'MaritalStatus'], drop_first=True)

# Splitting data into features and target variable
X = df.drop('Churn', axis=1)
y = df['Churn']

# Standardizing numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model selection and tuning (example with RandomForestClassifier)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Evaluating the model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy}")

# Additional evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Final thoughts and conclusions
print("\nFinal model details:")
print(best_model)
