import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file, abort
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Load the model (ensure the correct path)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'stock_dl_model.h5'))
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)

# Ensure the 'static/' directory exists
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'POWERGRID.NS'  # Default stock if none is entered

        # Define the start and end dates for stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()  # Use the current date instead of a hardcoded date

        try:
            # Download stock data
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                raise ValueError(f"No data found for stock: {stock}")
        except Exception as e:
            return render_template('index.html', error=f"Error fetching stock data: {e}")

        # Descriptive Data
        data_desc = df.describe()

        # Exponential Moving Averages
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Data splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Prepare data for prediction
        past_100_days = data_training.tail(100)
        final_df = past_100_days.append(data_testing, ignore_index=True)
        input_data = scaler.transform(final_df)  # Use transform instead of fit_transform to avoid data leakage

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)

        # Inverse scaling for predictions
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ema_chart_path = os.path.join(static_dir, "ema_20_50.png")
        fig1.savefig(ema_chart_path)
        plt.close(fig1)

        # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ema_chart_path_100_200 = os.path.join(static_dir, "ema_100_200.png")
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)

        # Plot 3: Prediction vs Original Trend
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        prediction_chart_path = os.path.join(static_dir, "stock_prediction.png")
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Save dataset as CSV
        csv_file_path = os.path.join(static_dir, f"{stock}_dataset.csv")
        df.to_csv(csv_file_path)

        # Return the rendered template with charts and dataset
        return render_template(
            'index.html',
            plot_path_ema_20_50=ema_chart_path,
            plot_path_ema_100_200=ema_chart_path_100_200,
            plot_path_prediction=prediction_chart_path,
            data_desc=data_desc.to_html(classes='table table-bordered'),
            dataset_link=os.path.basename(csv_file_path)
        )

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(static_dir, filename)
    if not os.path.exists(file_path):
        abort(404)  # Return a 404 error if the file does not exist
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
