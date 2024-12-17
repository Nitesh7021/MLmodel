from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# ======== 1. Load Models ========
# Model Directory
model_dir = r'C:\Users\nitin sharma\Downloads\project\model'

# Load pre-trained XGBoost models
try:
    with open(os.path.join(model_dir, 'xgb_open.pkl'), 'rb') as f:
        model_open = pickle.load(f)

    with open(os.path.join(model_dir, 'xgb_close.pkl'), 'rb') as f:
        model_close = pickle.load(f)
    print("Models loaded successfully!")

except Exception as e:
    print(f"Error loading models: {e}")
    raise SystemExit("Ensure 'xgb_open.pkl' and 'xgb_close.pkl' exist in the model folder.")

# ======== 2. Home Route ========
@app.route('/')
def home():
    return render_template('index.html')

# ======== 3. Predict Route ========
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check for file upload
        if 'csv_file' not in request.files:
            return jsonify({"error": "No file part. Please upload a file."})

        file = request.files['csv_file']

        if file.filename == '':
            return jsonify({"error": "No file selected. Please upload a CSV file."})

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Invalid file format. Only CSV files are allowed."})

        # ======== 4. Load and Validate CSV ========
        data = pd.read_csv(file)
        if not {'Open', 'Close', 'Date'}.issubset(data.columns):
            return jsonify({"error": "CSV file must contain 'Date', 'Open', and 'Close' columns."})

        # Preprocess data
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
        data.set_index('Date', inplace=True)

        # ======== 5. Create Features ========
        lags = 10
        for lag in range(1, lags + 1):
            data[f'Open_lag_{lag}'] = data['Open'].shift(lag)
            data[f'Close_lag_{lag}'] = data['Close'].shift(lag)

        # Moving Averages (5 and 10 days)
        data['Open_MA_5'] = data['Open'].rolling(window=5).mean()
        data['Close_MA_5'] = data['Close'].rolling(window=5).mean()
        data['Open_MA_10'] = data['Open'].rolling(window=10).mean()
        data['Close_MA_10'] = data['Close'].rolling(window=10).mean()

        # Volatility
        data['Open_volatility'] = data['Open'].rolling(window=5).std()
        data['Close_volatility'] = data['Close'].rolling(window=5).std()

        # Drop NaN values created by lagging
        data.dropna(inplace=True)

        # ======== 6. Predict Next 5 Days ========
        last_row = data.iloc[-1].copy()
        predictions_open, predictions_close = [], []

        for _ in range(5):
            input_features = np.array(
                [last_row[f'Open_lag_{i}'] for i in range(1, lags + 1)] +
                [last_row[f'Close_lag_{i}'] for i in range(1, lags + 1)] +
                [last_row['Open_MA_5'], last_row['Close_MA_5'],
                 last_row['Open_MA_10'], last_row['Close_MA_10'],
                 last_row['Open_volatility'], last_row['Close_volatility']]
            ).reshape(1, -1)

            next_open = model_open.predict(input_features)[0]
            next_close = model_close.predict(input_features)[0]

            predictions_open.append(next_open)
            predictions_close.append(next_close)

            # Update lag features for the next prediction
            for lag in range(lags, 1, -1):
                last_row[f'Open_lag_{lag}'] = last_row[f'Open_lag_{lag - 1}']
                last_row[f'Close_lag_{lag}'] = last_row[f'Close_lag_{lag - 1}']
            last_row['Open_lag_1'], last_row['Close_lag_1'] = next_open, next_close

        # ======== 7. Generate Graphs ========
        start_date = data.index[-1] + pd.Timedelta(days=1)
        dates = pd.date_range(start=start_date, periods=5, freq='D')

        # Graph 1: Predicted Prices
        plt.figure(figsize=(12, 6))
        plt.plot(dates, predictions_open, label='Predicted Open', marker='o', linestyle='--', color='orange')
        plt.plot(dates, predictions_close, label='Predicted Close', marker='o', linestyle='--', color='red')
        plt.title("Predicted Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf1.seek(0)
        img_base64_1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
        buf1.close()

        # Graph 2: Actual vs Predicted Prices
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Open'], label='Actual Open', color='blue')
        plt.plot(data.index, data['Close'], label='Actual Close', color='green')
        plt.plot(dates, predictions_open, label='Predicted Open', linestyle='--', color='orange')
        plt.plot(dates, predictions_close, label='Predicted Close', linestyle='--', color='red')
        plt.title("Actual vs Predicted Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        img_base64_2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
        buf2.close()

        # Prepare predictions for display
        predictions_text = list(zip(dates, predictions_open, predictions_close))
        return render_template(
            'index.html',
            predictions=predictions_text,
            img_base64_1=img_base64_1,
            img_base64_2=img_base64_2
        )

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"})

# ======== 8. Run Flask App ========
if __name__ == '__main__':
    app.run(debug=True)
