# Load dataset (assuming media_prediction_and_its_cost.csv is in the current directory)
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('media_prediction_and_its_cost.csv')

# Initial data exploration
print("Dataset structure and summary statistics:")
print(df.head())
print(df.describe())

# Handle missing values (if any)
df.fillna(df.median(), inplace=True)

# Splitting data into features and target variable
X = df.drop('Cost', axis=1)  # Assuming 'Cost' is the target variable
y = df['Cost']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and tuning (example with LinearRegression)
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse}")

# Additional evaluation metrics (if needed for regression)
# Example: R-squared score
r2_score = model.score(X_test, y_test)
print(f"R-squared score: {r2_score}")

# Final model details (coefficients, intercept, etc.)
print("\nFinal model details:")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
