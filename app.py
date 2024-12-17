
from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load models from the correct directory
model_dir = r'C:\Users\nitin sharma\Downloads\project\model'

# Load the pre-trained XGBoost models
with open(os.path.join(model_dir, 'xgb_open.pkl'), 'rb') as f:
    model_open = pickle.load(f)

with open(os.path.join(model_dir, 'xgb_close.pkl'), 'rb') as f:
    model_close = pickle.load(f)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'csv_file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['csv_file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and file.filename.endswith('.csv'):
        # Read CSV file
        data = pd.read_csv(file)
        
        # Ensure the data has the necessary columns
        if not {'Open', 'Close', 'Date'}.issubset(data.columns):
            return jsonify({"error": "Data must contain 'Date', 'Open', and 'Close' columns."})

        # Convert 'Date' column to datetime and set as index
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
        data.set_index('Date', inplace=True)

        # Create lag features
        lags = 10
        for lag in range(1, lags + 1):
            data[f'Open_lag_{lag}'] = data['Open'].shift(lag)
            data[f'Close_lag_{lag}'] = data['Close'].shift(lag)

        # Add moving averages and volatility
        data['Open_MA_5'] = data['Open'].rolling(window=5).mean()
        data['Close_MA_5'] = data['Close'].rolling(window=5).mean()
        data['Open_MA_10'] = data['Open'].rolling(window=10).mean()
        data['Close_MA_10'] = data['Close'].rolling(window=10).mean()

        data['Open_volatility'] = data['Open'].rolling(window=5).std()
        data['Close_volatility'] = data['Close'].rolling(window=5).std()

        # Drop rows with NaN values after creating lag features
        data = data.dropna()

        last_row = data.iloc[-1].copy()
        predictions_open = []
        predictions_close = []

        for _ in range(5):
            input_features = np.array(
                [last_row[f'Open_lag_{i}'] for i in range(1, lags + 1)] +
                [last_row[f'Close_lag_{i}'] for i in range(1, lags + 1)] +
                [last_row[f'Open_MA_5'], last_row[f'Close_MA_5'],
                 last_row[f'Open_MA_10'], last_row[f'Close_MA_10'],
                 last_row[f'Open_volatility'], last_row[f'Close_volatility']])
            input_features = input_features.reshape(1, -1)

            next_open = model_open.predict(input_features)[0]
            next_close = model_close.predict(input_features)[0]

            predictions_open.append(next_open)
            predictions_close.append(next_close)

            for lag in range(lags, 1, -1):
                last_row[f'Open_lag_{lag}'] = last_row[f'Open_lag_{lag - 1}']
                last_row[f'Close_lag_{lag}'] = last_row[f'Close_lag_{lag - 1}']
            last_row['Open_lag_1'] = next_open
            last_row['Close_lag_1'] = next_close

        start_date = data.index[-1] + pd.Timedelta(days=1)
        dates = pd.date_range(start=start_date, periods=5, freq='D')

        # First graph: Predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(dates, predictions_open, label='Predicted Open Prices', marker='o', linestyle='--', color='orange')
        plt.plot(dates, predictions_close, label='Predicted Close Prices', marker='o', linestyle='--', color='red')
        plt.title('Predicted Open and Close Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()

        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf1.seek(0)
        img_base64_1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
        buf1.close()

        # Second graph: Actual vs predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Open'], label='Actual Open Prices', color='blue')
        plt.plot(data.index, data['Close'], label='Actual Close Prices', color='green')
        plt.plot(dates, predictions_open, label='Predicted Open Prices', marker='o', linestyle='--', color='orange')
        plt.plot(dates, predictions_close, label='Predicted Close Prices', marker='o', linestyle='--', color='red')
        plt.title('Actual and Predicted Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()

        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        img_base64_2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
        buf2.close()

        predictions_text = list(zip(dates, predictions_open, predictions_close))
        return render_template(
            'index.html',
            predictions=predictions_text,
            img_base64_1=img_base64_1,
            img_base64_2=img_base64_2
        )

    return jsonify({"error": "Invalid file format. Please upload a CSV file."})

if __name__ == '__main__':
    app.run(debug=True)
