import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from neuralforecast.losses.numpy import mae, mse, mape

TICKERS = ["SPY", "SMH", "IBB", "GDX", "IYR", "KBE"] 
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    close_prices = stock_data[['Close']].reset_index()
    close_prices = close_prices.rename(columns={"Date": "ds", "Close": "y"})
    close_prices['ds'] = pd.to_datetime(close_prices['ds'])
    close_prices = close_prices.dropna()
    return close_prices


def fill_missing(df, start_date, end_date):
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = df.set_index('ds').reindex(full_date_range).reset_index()
    df.columns = ['ds', 'y']
    df['y'] = df['y'].fillna(method='ffill').fillna(method='bfill')
    return df


def prepare_data(ticker):
    train_start_date = "2024-01-01"
    train_end_date = "2024-06-08"

    test_start_date = "2024-06-01"
    test_end_date = "2024-06-08"

    train_df = load_stock_data(ticker, train_start_date, train_end_date)
    test_df = load_stock_data(ticker, test_start_date, test_end_date)

    train_df = fill_missing(train_df, train_start_date, train_end_date)
    test_df = fill_missing(test_df, test_start_date, test_end_date)
    return train_df, test_df


def load_csv(csv_path):
    forecasts = pd.read_csv(csv_path)
    forecasts["ds"] = pd.to_datetime(forecasts["ds"], errors="coerce")
    return forecasts


def eval_model(test_df, forecasts):
    mae_value = mae(test_df['y'], forecasts['TimeLLM'])
    mse_value = mse(test_df['y'], forecasts['TimeLLM'])
    mape_value = mape(test_df['y'], forecasts['TimeLLM'])

    return mae_value, mse_value, mape_value


def plot_forecast(train_df, forecasts, mae_value, mse_value, mape_value, file_path):
    plt.figure(figsize=(12, 6))

    plt.plot(train_df["ds"], train_df["y"], label="Actual", color="#2496ED", linestyle="-")
    plt.plot(forecasts["ds"], forecasts["TimeLLM"], label="Prediction (TimeLLM)", color="#FF6A6A", linestyle="-")

    plt.axvline(x=forecasts["ds"][0], color="gray", linestyle=":")

    plt.title("Train and Prediction Visualization", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=45)

    metrics_text = f"MAE: {mae_value:.2f}\nMSE: {mse_value:.2f}\nMAPE: {mape_value:.2f}"
    plt.plot([], [], ' ', label=metrics_text)
    plt.legend(fontsize=12)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(file_path)


def make_radar_chart(tickers, mae_values, mse_values, mape_values, png_path):
    num_vars = 6
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axs = plt.subplots(1, 3, figsize=(12, 5), subplot_kw=dict(polar=True))

    # MAE Chart
    mae_values = mae_values[:num_vars] + mae_values[:1]
    axs[0].plot(angles, mae_values, color='blue', linewidth=2)
    axs[0].fill(angles, mae_values, color='blue', alpha=0.25)
    axs[0].set_yticks([])
    axs[0].set_xticks(angles[:-1])
    axs[0].set_xticklabels(tickers[:num_vars])
    axs[0].set_title("MAE score", fontsize=12)

    outer_radius = max(mae_values) 
    axs[0].plot(angles[:-1] + [angles[0]], [outer_radius] * len(angles), color='gray', linestyle='-', linewidth=1.0)
    for angle in angles[:-1]:
        axs[0].plot([angle, angle], [0, outer_radius], color='gray', linestyle='--', linewidth=1.0)


    # MSE Chart
    mse_values = mse_values[:num_vars] + mse_values[:1]
    axs[1].plot(angles, mse_values, color='green', linewidth=2)
    axs[1].fill(angles, mse_values, color='green', alpha=0.25)
    axs[1].set_yticks([])
    axs[1].set_xticks(angles[:-1])
    axs[1].set_xticklabels(tickers[:num_vars])
    axs[1].set_title("MSE score", fontsize=12)

    outer_radius = max(mse_values) 
    axs[1].plot(angles[:-1] + [angles[0]], [outer_radius] * len(angles), color='gray', linestyle='-', linewidth=1.0)
    for angle in angles[:-1]:
        axs[1].plot([angle, angle], [0, outer_radius], color='gray', linestyle='--', linewidth=0.8)


    # MAPE Chart
    mape_values = mape_values[:num_vars] + mape_values[:1]
    axs[2].plot(angles, mape_values, color='red', linewidth=2)
    axs[2].fill(angles, mape_values, color='red', alpha=0.25)
    axs[2].set_yticks([])
    axs[2].set_xticks(angles[:-1])
    axs[2].set_xticklabels(tickers[:num_vars])
    axs[2].set_title("MAPE score", fontsize=12)

    outer_radius = max(mape_values)
    axs[2].plot(angles[:-1] + [angles[0]], [outer_radius] * len(angles), color='gray', linestyle='-', linewidth=1.0)
    for angle in angles[:-1]:
        axs[2].plot([angle, angle], [0, outer_radius], color='gray', linestyle='--', linewidth=0.8)

    for ax in axs:
        ax.spines['polar'].set_visible(False)
        ax.set_theta_offset(np.pi / 6)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(png_path)


if __name__ == "__main__":
    mae_values = []
    mse_values = []
    mape_values = []

    for ticker in TICKERS:
        train_df, test_df = prepare_data(ticker)
        forecasts = load_csv(csv_path=f"./result/{ticker}.csv")
        mae_value, mse_value, mape_value = eval_model(test_df, forecasts)
        mae_values.append(mae_value)
        mse_values.append(mse_value)
        mape_values.append(mape_value)
        plot_forecast(train_df, forecasts, mae_value, mse_value, mape_value, file_path=f"./result/result_{ticker}.png")
    
    make_radar_chart(TICKERS, mae_values, mse_values, mape_values, png_path="./result/radar_charts.png")