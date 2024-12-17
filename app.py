from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ======== 1. Load Models Safely ========
# Define the directory containing the models
model_dir = os.path.join(r'C:', 'Users', 'nitin sharma', 'Downloads', 'project', 'model')

try:
    # Load the pre-trained XGBoost models
    model_open_path = os.path.join(model_dir, 'xgb_open.pkl')
    model_close_path = os.path.join(model_dir, 'xgb_close.pkl')

    with open(model_open_path, 'rb') as f:
        model_open = pickle.load(f)

    with open(model_close_path, 'rb') as f:
        model_close = pickle.load(f)

    print("Models loaded successfully!")

except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    raise SystemExit(f"Model files not found. Ensure 'xgb_open.pkl' and 'xgb_close.pkl' are in the directory: {model_dir}")
except Exception as e:
    print(f"Unexpected error: {e}")
    raise SystemExit("An unexpected error occurred while loading models.")

# ======== 2. Home Route ========
@app.route('/')
def home():
    return render_template('index.html')

# ======== 3. Predict Route ========
@app.route('/predict', methods=['POST'])
def predict():
    if 'csv_file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['csv_file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and file.filename.endswith('.csv'):
        try:
            # Read CSV file
            data = pd.read_csv(file)

            # Ensure the data has necessary columns
            if not {'Open', 'Close', 'Date'}.issubset(data.columns):
                return jsonify({"error": "Data must contain 'Date', 'Open', and 'Close' columns."})

            # Preprocess data
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

            data = data.dropna()
            last_row = data.iloc[-1].copy()

            # Predict next 5 days
            predictions_open, predictions_close = [], []
            for _ in range(5):
                input_features = np.array(
                    [last_row[f'Open_lag_{i}'] for i in range(1, lags + 1)] +
                    [last_row[f'Close_lag_{i}'] for i in range(1, lags + 1)] +
                    [last_row['Open_MA_5'], last_row['Close_MA_5'],
                     last_row['Open_MA_10'], last_row['Close_MA_10'],
                     last_row['Open_volatility'], last_row['Close_volatility']]
                ).reshape(1, -1)

                # Make predictions
                next_open = model_open.predict(input_features)[0]
                next_close = model_close.predict(input_features)[0]
                predictions_open.append(next_open)
                predictions_close.append(next_close)

                # Update lag features
                for lag in range(lags, 1, -1):
                    last_row[f'Open_lag_{lag}'] = last_row[f'Open_lag_{lag - 1}']
                    last_row[f'Close_lag_{lag}'] = last_row[f'Close_lag_{lag - 1}']
                last_row['Open_lag_1'] = next_open
                last_row['Close_lag_1'] = next_close

            # Prepare graph dates
            start_date = data.index[-1] + pd.Timedelta(days=1)
            dates = pd.date_range(start=start_date, periods=5, freq='D')

            # Plot predictions
            plt.figure(figsize=(12, 6))
            plt.plot(dates, predictions_open, label='Predicted Open Prices', marker='o', linestyle='--', color='orange')
            plt.plot(dates, predictions_close, label='Predicted Close Prices', marker='o', linestyle='--', color='red')
            plt.title('Predicted Open and Close Stock Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            return render_template(
                'index.html',
                predictions=list(zip(dates, predictions_open, predictions_close)),
                img_base64=img_base64
            )

        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"})
    else:
        return jsonify({"error": "Invalid file format. Please upload a CSV file."})

# ======== 4. Run the Flask App ========
if __name__ == '__main__':
    app.run(debug=True)
