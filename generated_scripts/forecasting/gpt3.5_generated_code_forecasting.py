# Load dataset (assuming retail_sales.csv is in the current directory)
import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the dataset
df = pd.read_csv('retail_sales.csv')

# Initial data exploration
print("Dataset structure and summary statistics:")
print(df.head())
print(df.describe())

# Renaming columns for Prophet
df = df.rename(columns={'Date': 'ds', 'Sales': 'y'})

# Model selection and training (example with Prophet)
model = Prophet()
model.fit(df)

# Forecasting future periods
future = model.make_future_dataframe(periods=365)  # Forecasting for 1 year (365 days) into the future
forecast = model.predict(future)

# Evaluating the model using RMSE
y_true = df['y']
y_pred = forecast['yhat'].tail(len(y_true))  # Taking predictions for the original time series length
rmse = sqrt(mean_squared_error(y_true, y_pred))
print(f"\nRoot Mean Squared Error (RMSE): {rmse}")
