import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
# Library load
# Basic libraries
import time
import json
import requests  # for function calls like get_stock_price, if they use external
import warnings
import os
import requests
import asyncio
import configparser
import asyncio
# data analysis
import numpy as np
import pandas as pd
# financial libs
import yfinance as yf # terminal install -> pip install yfinance
import talib # terminal install -> pip install TA-Lib
# OpenAI libs
import openai # terminal install ->  pip install openai
from openai import OpenAI  
from agents import Runner, Agent,function_tool, items # terminal install -> pip 
from agents import (Agent, Runner, FunctionTool, InputGuardrail, GuardrailFunctionOutput,
                    handoff,  InputGuardrailTripwireTriggered, RunConfig, ModelSettings)
from agents.exceptions import InputGuardrailTripwireTriggered
from agents.extensions import handoff_filters
from pydantic import BaseModel
from pytorch_lightning.callbacks import EarlyStopping

from darts.metrics import mape

key_file_name = 'api_key'
path = 'C:/Users/phann/Documents/LLM_mentor/LLM_TRADE'  # use your path to where you
config = configparser.ConfigParser()
config.read(path+'/'+ key_file_name) 
api_key = config['openai']['api_key'] 
os.environ["OPENAI_API_KEY"] = api_key # required by OpenAI agents
fmp_key = config['financialmodelingprep']['api_key'] 
warnings.filterwarnings('ignore')

# Hàm lấy dữ liệu lịch sử (từ mã bạn cung cấp)
def get_historical_data(symbol: str) -> pd.DataFrame:
    """
    Fetch historical closing prices from FMP API.
    Returns DataFrame with 'date' and 'close' columns, ensuring daily frequency.
    """
    start_date = "2022-06-01"  # Fetch from early April to get >30 days
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&apikey={fmp_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "historical" not in data or not data["historical"]:
            return pd.DataFrame()
        df = pd.DataFrame(data["historical"])[["date", "close"]]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(date_range, method='ffill')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

# Hàm chính để lấy dữ liệu, huấn luyện và vẽ biểu đồ
def forecast_and_plot(symbol: str, forecast_horizon: int = 100):
    try:
        # Bước 1: Lấy dữ liệu lịch sử
        df = get_historical_data(symbol)
        if df.empty:
            print(f"No data available for symbol: {symbol}")
            return
        
        print(f"Data range: {df['date'].min()} to {df['date'].max()}")
        
        # Tính toán cả SMA50 và SMA200
        sma_period_50 = 50
        sma_period_200 = 200
        df['sma_50'] = talib.SMA(df['close'], timeperiod=sma_period_50)
        df['sma_200'] = talib.SMA(df['close'], timeperiod=sma_period_200)
        
        # Xóa các dòng có giá trị NaN
        df = df.dropna(subset=['sma_50', 'sma_200'])
        
        # Bước 2: Chuẩn bị TimeSeries với cả SMA50 và SMA200
        series = TimeSeries.from_dataframe(
            df,
            time_col='date',
            value_cols=['sma_50', 'sma_200'],
            fill_missing_dates=True,
            freq='D'
        )
        
        # Chuẩn hóa dữ liệu
        scaler = Scaler()
        series_scaled = scaler.fit_transform(series)
        
        # Tách tập huấn luyện và kiểm tra
        train, val = series_scaled.split_after(0.8)
        
        print(f"Training data range: {train.time_index[0]} to {train.time_index[-1]}")
        print(f"Validation data range: {val.time_index[0]} to {val.time_index[-1]}")
        
        # Bước 3: Khởi tạo và huấn luyện mô hình N-BEATS
        model = NBEATSModel(
            input_chunk_length=60,  # Tăng để bắt xu hướng dài hạn
            output_chunk_length=forecast_horizon,
            n_epochs=10,
            random_state=42)

        model.fit(train, val_series=val, verbose=True)

        # Save model weights
        model_dir = "saved_models_NBEATS"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"nbeats_{symbol}.pt")
        model.save(model_path)
        print(f"Model weights saved to {model_path}")
        
        # Bước 4: Dự báo trên tập val
        val_forecast_scaled = model.predict(n=len(val), series=train)
        val_forecast = scaler.inverse_transform(val_forecast_scaled)
        val_actual = scaler.inverse_transform(val)
        
        # Tính MAPE trên tập val
        error = mape(val_actual, val_forecast)
        print(f"MAPE on validation set: {error:.2f}%")
        
        # Bước 5: Dự báo tương lai - Sử dụng cả train và validation data
        last_point = series_scaled
        # Kết hợp train và validation để có dữ liệu mới nhất
        forecast_scaled = model.predict(n=forecast_horizon, series=series_scaled)
        forecast = scaler.inverse_transform(forecast_scaled)
        
        # Bước 6: Chuẩn bị dữ liệu để vẽ biểu đồ
        historical_dates = series.time_index
        historical_values_sma50 = series.values()[:, 0]  # Lấy giá trị SMA50
        historical_values_sma200 = series.values()[:, 1]  # Lấy giá trị SMA200
        
        val_dates = val.time_index
        val_values_sma50 = val_actual.values()[:, 0]
        val_values_sma200 = val_actual.values()[:, 1]
        
        val_forecast_values_sma50 = val_forecast.values()[:, 0]
        val_forecast_values_sma200 = val_forecast.values()[:, 1]
        
        forecast_dates = forecast.time_index
        forecast_values_sma50 = forecast.values()[:, 0]
        forecast_values_sma200 = forecast.values()[:, 1]
        
        # Bước 7: Vẽ biểu đồ cho cả SMA50 và SMA200
        plt.figure(figsize=(15, 8))
        
        # Vẽ dữ liệu lịch sử
        plt.plot(historical_dates, historical_values_sma50, label='Historical SMA50', color='blue', alpha=0.7)
        plt.plot(historical_dates, historical_values_sma200, label='Historical SMA200', color='purple', alpha=0.7)
        
        # Vẽ dữ liệu validation
        plt.plot(val_dates, val_values_sma50, label='Validation SMA50', color='green', alpha=0.7)
        plt.plot(val_dates, val_values_sma200, label='Validation SMA200', color='orange', alpha=0.7)
        
        # Vẽ dự báo validation
        plt.plot(val_dates, val_forecast_values_sma50, label='Validation Forecast SMA50', color='green', linestyle='--')
        plt.plot(val_dates, val_forecast_values_sma200, label='Validation Forecast SMA200', color='orange', linestyle='--')
        
        # Vẽ dự báo tương lai
        plt.plot(forecast_dates, forecast_values_sma50, label='Future Forecast SMA50', color='red', linestyle='--')
        plt.plot(forecast_dates, forecast_values_sma200, label='Future Forecast SMA200', color='brown', linestyle='--')
        
        plt.title(f'Stock Price Forecast for {symbol} (MAPE: {error:.2f}%)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{symbol}_forecast.png')
        
        # In kết quả dự báo tương lai
        print(f"\nForecast for {symbol} (next {forecast_horizon} days):")
        print("Date\t\tSMA50 Forecast\tSMA200 Forecast")
        print("-" * 50)
        for i in range(len(forecast_dates)):
            print(f"{forecast_dates[i].date()}\t${forecast_values_sma50[i]:.2f}\t\t${forecast_values_sma200[i]:.2f}")
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")


# Chạy ví dụ với một mã cổ phiếu
if __name__ == "__main__":
    symbol = "NVDA"  # Ví dụ với cổ phiếu Apple
    forecast_and_plot(symbol, forecast_horizon=100)