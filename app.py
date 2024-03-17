import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras 
from keras.models import load_model
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

# Load data
company = 'AAPL'
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2023, 3, 2)

try:
    data = yf.download(company, start=start, end=end)
except Exception as e:
    print(f"Failed to download data for {company}: {e}")
    exit()

# Check if data was downloaded successfully
if len(data) == 0:
    print(f"No data downloaded for {company}")
    exit()

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

# Load the saved model
loaded_model = load_model("stock_prediction_model.h5")

''' Test the Model Accuracy on Existing Data'''

# Load Test Data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make predictions on test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = loaded_model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the Test Predictions with extended x-axis
plt.plot(test_data.index, actual_prices, color="black", label=f"Actual {company} Prices")
plt.plot(test_data.index, predicted_prices, color="green", label=f"Predicted {company} Prices")

# Predict next day
real_data = [model_inputs[len(model_inputs)-prediction_days:len(model_inputs), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

next_day_prediction = loaded_model.predict(real_data)
next_day_prediction = scaler.inverse_transform(next_day_prediction)

# Extend the x-axis
extended_dates = pd.date_range(start=test_start, end=test_end + dt.timedelta(days=1))

# Plot next day's prediction
plt.plot(extended_dates[-1], next_day_prediction, 'ro', label=f"Predicted {company} Price Next Day")  

# Annotate next day's prediction value on the graph
plt.annotate(f'Next Day Prediction: {next_day_prediction[0][0]}',
             xy=(extended_dates[-1], next_day_prediction),
             xytext=(extended_dates[-1] + dt.timedelta(days=2), next_day_prediction + 10),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

print(f"Next Day Prediction: {next_day_prediction[0][0]}")
