import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/retail_sales.csv'  # Update with your dataset path
data = pd.read_csv(file_path)

# Ensure the data is in time series format
# Assuming the dataset has a 'Date' column and a 'Sales' column
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Fit the ARIMA model
model = sm.tsa.ARIMA(train['Sales'], order=(5, 1, 0))  # Adjust the order (p, d, q) as needed
model_fit = model.fit(disp=0)

# Forecast
forecast = model_fit.forecast(steps=len(test))[0]

# Evaluate the model
rmse = sqrt(mean_squared_error(test['Sales'], forecast))

print("Root Mean Squared Error:", rmse)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train['Sales'], label='Train')
plt.plot(test['Sales'], label='Test')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.show()
