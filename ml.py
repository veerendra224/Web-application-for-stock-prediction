import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of the next closing value
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=25, batch_size=32)
    return model

def save_model(model, filename):
    model.save(filename)

def load_model(filename):
    return keras.models.load_model(filename)

def predict_next_day(model, real_data, scaler):
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction

def test_model_accuracy(model, test_data, target_attribute, scaler, prediction_days):
    actual_prices = test_data[target_attribute].values

    total_dataset = pd.concat((data[target_attribute], test_data[target_attribute]), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)+1):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    errors = actual_prices - predicted_prices
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2
    mae = np.mean(absolute_errors)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    accuracy_threshold = 0.05
    accurate_predictions = np.sum(np.abs(errors) < accuracy_threshold)
    accuracy_percentage = (accurate_predictions / len(actual_prices)) * 100

    return mae, mse, rmse, accuracy_percentage, actual_prices, predicted_prices


# Load data
company = 'AAPL'
start = dt.datetime(2012, 1, 1)
end = dt.datetime.today()

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
scalers = {}
models = {}
for attribute in ['Open', 'High','Open', 'Close']:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[attribute].values.reshape(-1, 1))
    scalers[attribute] = scaler

    prediction_days = 60

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = np.array(y_train)
    y_train = y_train.reshape(-1, 1)

    input_shape = (x_train.shape[1], 1)
    model = build_model(input_shape)
    model = train_model(model, x_train, y_train)
    models[attribute] = model

    save_model(model, f"stock_prediction_model_{attribute}.h5")

    print(f"Model trained and saved for attribute: {attribute}")

    # Test the model accuracy
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()

    test_data = yf.download(company, start=test_start, end=test_end)
    mae, mse, rmse, accuracy_percentage, actual_prices, predicted_prices = test_model_accuracy(model, test_data, attribute, scaler, prediction_days)

    print(f"Attribute: {attribute}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Accuracy Percentage: {accuracy_percentage}%")

    # Plot the Test Predictions
    plt.plot(actual_prices, color="black", label=f"Actual {company} {attribute} Prices")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} {attribute} Prices")
    plt.title(f"{company} {attribute} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} {attribute} Share Price')
    plt.legend()
    plt.show()

    # Predict next day
    real_data = [scaled_data[len(scaled_data)+1 - prediction_days:len(scaled_data+1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = predict_next_day(model, real_data, scaler)
    print(f"Next day prediction for {attribute}: {prediction}")
