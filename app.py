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

# ======== 1. Load Models Safely ========
# Relative Path: Ensure compatibility across systems
model_dir = os.path.join(os.getcwd(), 'model')

try:
    # Load the pre-trained models
    with open(os.path.join(model_dir, 'xgb_open.pkl'), 'rb') as f:
        model_open = pickle.load(f)

    with open(os.path.join(model_dir, 'xgb_close.pkl'), 'rb') as f:
        model_close = pickle.load(f)

    print("Models loaded successfully!")

except Exception as e:
    print(f"Error loading models: {e}")
    raise SystemExit("Model files missing or corrupt. Ensure 'xgb_open.pkl' and 'xgb_close.pkl' are in the 'model' folder.")

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

        # Load the CSV data
        data = pd.read_csv(file)
        if not {'Open', 'Close', 'Date'}.issubset(data.columns):
            return jsonify({"error": "CSV file must contain 'Date', 'Open', and 'Close' columns."})

        # ======== 4. Preprocess Data ========
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
        data.set_index('Date', inplace=True)

        # Lag Features (10 days)
        lags = 10
        for lag in range(1, lags + 1):
            data[f'Open_lag_{lag}'] = data['Open'].shift(lag)
            data[f'Close_lag_{lag}'] = data['Close'].shift(lag)

        # Moving Averages and Volatility
        data['Open_MA_5'] = data['Open'].rolling(window=5).mean()
        data['Close_MA_5'] = data['Close'].rolling(window=5).mean()
        data['Open_volatility'] = data['Open'].rolling(window=5).std()
        data = data.dropna()

        # ======== 5. Predict Future Prices ========
        last_row = data.iloc[-1].copy()
        predictions_open, predictions_close = [], []

        for _ in range(5):
            input_features = np.array(
                [last_row[f'Open_lag_{i}'] for i in range(1, lags + 1)] +
                [last_row[f'Close_lag_{i}'] for i in range(1, lags + 1)] +
                [last_row['Open_MA_5'], last_row['Close_MA_5'], last_row['Open_volatility']]
            ).reshape(1, -1)

            next_open = model_open.predict(input_features)[0]
            next_close = model_close.predict(input_features)[0]
            predictions_open.append(next_open)
            predictions_close.append(next_close)

            # Shift values to simulate new row
            for lag in range(lags, 1, -1):
                last_row[f'Open_lag_{lag}'] = last_row[f'Open_lag_{lag - 1}']
                last_row[f'Close_lag_{lag}'] = last_row[f'Close_lag_{lag - 1}']
            last_row['Open_lag_1'], last_row['Close_lag_1'] = next_open, next_close

        # ======== 6. Create Graphs ========
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
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf1.seek(0)
        img_base64_1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
        buf1.close()

        # Graph 2: Actual vs Predicted
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Open'], label='Actual Open', color='blue')
        plt.plot(data.index, data['Close'], label='Actual Close', color='green')
        plt.plot(dates, predictions_open, label='Predicted Open', linestyle='--', color='orange')
        plt.plot(dates, predictions_close, label='Predicted Close', linestyle='--', color='red')
        plt.title("Actual vs Predicted Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        img_base64_2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
        buf2.close()

        # ======== 7. Return Predictions ========
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
