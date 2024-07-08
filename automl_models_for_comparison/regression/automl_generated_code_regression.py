import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'media_prediction_and_its_cost.csv'  # Update with your dataset path
data = pd.read_csv(file_path)

# Drop columns that are not useful for prediction (if any)
# For demonstration, let's assume all columns are useful
# data = data.drop(['unnecessary_column'], axis=1)

# Split the dataset into features and target variable
# Assuming the last column is the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the AutoML regressor using TPOT
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42, n_jobs=-1)

# Train the AutoML model
tpot.fit(X_train, y_train)

# Predict the test set results
y_pred = tpot.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
