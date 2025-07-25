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

from darts.models import RNNModel, NBEATSModel, TransformerModel, TCNModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Import tools from tool.py
## Retrieve API Keys and start OpenAI session
#  retrieve keys:
key_file_name = 'api_key'
path = r'C:\Users\phann\Documents\LLM_mentor\LLM_TRADE'  # use your path to where you
config = configparser.ConfigParser()
config.read(path+'/'+ key_file_name) 
api_key = config['openai']['api_key'] 
os.environ["OPENAI_API_KEY"] = api_key # required by OpenAI agents
fmp_key = config['financialmodelingprep']['api_key'] 
warnings.filterwarnings('ignore')

try:
    from darts import TimeSeries
    from darts.models import NBEATSModel
except ImportError as e:
    print(f"Failed to import Darts: {e}")
    raise
import nest_asyncio
import torch

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

# FMP API key (replace with your actual key)

# Tool to fetch historical data
def get_historical_data(symbol: str) -> pd.DataFrame:
    """
    Fetch historical closing prices from FMP API.
    Returns DataFrame with 'date' and 'close' columns, ensuring daily frequency.
    """
    start_date = "2024-06-01"  # Fetch from early April to get >30 days
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&apikey={fmp_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "historical" not in data or not data["historical"]:
            return pd.DataFrame()
        df = pd.DataFrame(data["historical"])[["date", "close"]]
        df["date"] = pd.to_datetime(df["date"])
        # Set date as index
        df.set_index("date", inplace=True)
        # Create a complete date range (daily, including weekends)
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        # Reindex to fill missing dates, forward-filling prices
        df = df.reindex(date_range, method='ffill')
        # Reset index to have 'date' as a column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

def get_rsi_data(symbol: str, days: int = 14) -> dict:
    df = get_historical_data(symbol)
    if df.empty:
        return {"error": f"No historical data for symbol: {symbol}"}
    close_np = df["close"].to_numpy(dtype='float64')
    rsi = talib.RSI(close_np, timeperiod=days)
    return {"symbol": symbol.upper(), "RSI": round(rsi[-1], 2)}

@function_tool
def get_stock_company_info(symbol: str) -> dict:
    """
    Fetch company's basic information using FMP API.
    """
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={fmp_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        info = response.json()
        if not info:
            return {"error": f"No data found for symbol: {symbol}"}
        return {
            "symbol": symbol.upper(),
            "price": info[0]["price"],
            "beta": info[0]["beta"],
            "description": info[0]["description"],
            "sector": info[0]["sector"],
            "industry": info[0]["industry"],
            "dcf_valuation": info[0]["dcf"]
        }
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}

@function_tool
def get_stock_technical(symbol: str) -> dict:
    """
    Calculate technical indicators (SMA, RSI, MACD) from FMP API historical data.
    """
    try:
        df = get_historical_data(symbol)
        if df.empty:
            return {"error": f"No historical data for symbol: {symbol}"}
        close_np = df["close"].to_numpy(dtype='float64')
        if len(close_np) < 200:
            return {"error": f"Not enough data to compute indicators for {symbol} (need at least 200 days)"}
        
        # Calculate indicators
        sma_50 = talib.SMA(close_np, timeperiod=50)
        sma_200 = talib.SMA(close_np, timeperiod=200)
        rsi = talib.RSI(close_np, timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(close_np, fastperiod=12, slowperiod=26, signalperiod=9)
        
        latest_sma_50 = sma_50[-1]
        latest_sma_200 = sma_200[-1]
        latest_rsi = rsi[-1]
        last_price = close_np[-1]
        latest_macd = macd[-1]
        latest_macdsignal = macdsignal[-1]
        latest_macdhist = macdhist[-1]
        
        # Calculate 6-month momentum
        if len(close_np) >= 126:  # Approx. 6 months
            roc_6m = close_np[-1] / close_np[-126] - 1
        else:
            roc_6m = None
        
        if np.isnan(latest_sma_50) or np.isnan(latest_sma_200) or np.isnan(latest_rsi) or np.isnan(latest_macd):
            return {"error": f"Indicator values not ready"}
        
        return {
            "symbol": symbol.upper(),
            "last_price": round(last_price, 2),
            "SMA_50d": round(latest_sma_50, 2),
            "SMA_200d": round(latest_sma_200, 2),
            "RSI": round(latest_rsi, 2),
            "MACD": round(latest_macd, 2),
            "MACD_signal": round(latest_macdsignal, 2),
            "MACD_hist": round(latest_macdhist, 2),
            "6_month_momentum": round(roc_6m, 2) if roc_6m is not None else None
        }
    except Exception as e:
        return {"error": f"Unhandled error for symbol {symbol}: {str(e)}"}

def get_model_path(symbol: str, model_type: str) -> str:
    """Generate a unique path for saving/loading model weights."""
    model_dir = r"C:\Users\phann\Documents\LLM_mentor\LLM_TRADE\saved_models_NBEATS"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Use .pt extension for NBEATS model, .pkl for others
    extension = ".pt" if model_type == "nbeats" else ".pkl"
    return os.path.join(model_dir, f"{model_type}_{symbol.lower()}{extension}")

@function_tool
def get_stock_forecast(symbol: str, horizon: int = 5, model_type: str = 'nbeats') -> dict:
    try:
        df = get_historical_data(symbol)
        if df.empty:
            return {"error": f"No historical data for symbol: {symbol}"}
        
        # Calculate SMA_50
        sma_period = 50
        df['sma_50'] = talib.SMA(df['close'], timeperiod=sma_period)
        df = df.dropna(subset=['sma_50'])
        
        # Create TimeSeries
        series = TimeSeries.from_dataframe(df, 'date', 'sma_50', fill_missing_dates=True, freq='D')
        
        # Split data
        train, val = series.split_after(0.8)
        
        # Select model
        input_chunk_length = 30
        model_path = get_model_path(symbol, model_type)
        
        # Try to load existing model
        if os.path.exists(model_path):
            try:
                if model_type == 'lstm':
                    model = RNNModel.load(model_path)
                elif model_type == 'nbeats':
                    model = NBEATSModel.load(model_path)
                elif model_type == 'transformer':
                    model = TransformerModel.load(model_path)
                elif model_type == 'tcn':
                    model = TCNModel.load(model_path)
                else:
                    return {"error": f"Invalid model type: {model_type}. Allowed: lstm, nbeats, transformer, tcn"}
            except Exception as e:
                print(f"Error loading model: {e}. Will train new model.")
                model = None
        else:
            model = None
            
        # If model doesn't exist or couldn't be loaded, create and train new one
        if model is None:
            # Scale the data
            scaler = Scaler()
            train_scaled = scaler.fit_transform(train)
            val_scaled = scaler.transform(val)
            
            if model_type == 'lstm':
                model = RNNModel(model='LSTM', input_chunk_length=input_chunk_length, training_length=18, 
                               output_chunk_length=horizon, hidden_dim=25, n_rnn_layers=2)
            elif model_type == 'nbeats':
                model = NBEATSModel(
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=horizon,
                    n_epochs=100,
                    random_state=42
                )
            elif model_type == 'transformer':
                model = TransformerModel(input_chunk_length=input_chunk_length, output_chunk_length=horizon, 
                                       d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=3)
            elif model_type == 'tcn':
                model = TCNModel(input_chunk_length=input_chunk_length, output_chunk_length=horizon, 
                               kernel_size=3, num_filters=32)
            else:
                return {"error": f"Invalid model type: {model_type}. Allowed: lstm, nbeats, transformer, tcn"}
            
            # Train model
            model.fit(train_scaled, val_series=val_scaled, verbose=True)
            try:
                model.save(model_path)
                print(f"Model saved to {model_path}")
            except Exception as e:
                print(f"Warning: Could not save model: {e}")
        
        # Forecast
        forecast = model.predict(n=horizon, series=series)
        predicted_sma = forecast.values().flatten().tolist()
        
        return {"symbol": symbol.upper(), "forecast_sma_50": [round(sma, 2) for sma in predicted_sma]}
    except Exception as e:
        return {"error": f"Error forecasting for symbol {symbol} with model {model_type}: {str(e)}"}

@function_tool
def RSI_analysis(symbol: str, days: int = 14) -> dict:
    df = get_historical_data(symbol)
    if df.empty:
        return {"error": f"No historical data for symbol: {symbol}"}
    close_np = df["close"].to_numpy(dtype='float64')
    rsi = talib.RSI(close_np, timeperiod=days)
    
    current_rsi = rsi[-1]
    previous_rsi = rsi[-2]
    
    # Tính toán xu hướng RSI
    rsi_trend = current_rsi - previous_rsi
    
    # Phân tích tín hiệu
    signals = []
    
    # Tín hiệu quá mua/quá bán
    if current_rsi > 80:
        signals.append("Overbought - SELL")
    elif current_rsi < 20:
        signals.append("Oversold - BUY")
    
    # Tín hiệu xu hướng
    if 40 <= previous_rsi <= 50 and current_rsi < 40:
        signals.append("RSI down from neutral - SELL")
    elif 50 <= previous_rsi <= 60 and current_rsi > 60:
        signals.append("RSI up from neutral - BUY")
    
    # Tín hiệu vùng lợi nhuận
    if previous_rsi >60 and 70 <= current_rsi <= 80:
        signals.append("Profit zone - Consider closing Buy")
    if previous_rsi < 40 and 20 <= current_rsi <= 30:
        signals.append("Profit zone - Consider closing SELL")
    
    return {
        "symbol": symbol.upper(),
        "RSI": round(current_rsi, 2),
        "RSI_trend": round(rsi_trend, 2),
        "signals": signals if signals else ["Neutral - HOLD"]
    }

@function_tool
def get_stock_macd_rsi(symbol: str) -> dict:
    """
    Calculate MACD and RSI indicators and detect trading signals.
    Returns stock price, MACD, RSI, and signals.
    """
    try:
        df = get_historical_data(symbol)
        if df.empty:
            return {"error": f"No historical data for symbol: {symbol}"}
        
        prices = df["close"].to_numpy(dtype='float64')[-26:]
        if len(prices) < 26:
            return {"error": f"Not enough data to compute indicators for {symbol} (need at least 26 days)"}
        
        # Calculate MACD
        macd, macdsignal, macdhist = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
        # Calculate RSI
        rsi = talib.RSI(prices, timeperiod=14)
        
        # Detect signals
        signals = [None] * len(macd)
        for i in range(1, len(macd)):
            if not np.isnan(macd[i]) and not np.isnan(macdsignal[i]) and not np.isnan(rsi[i]):
                if macd[i-1] < macdsignal[i-1] and macd[i] > macdsignal[i] and rsi[i] < 70:
                    signals[i] = 'BUY'
                elif macd[i-1] > macdsignal[i-1] and macd[i] < macdsignal[i] and rsi[i] > 30:
                    signals[i] = 'SELL'
        
        # Generate plot
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(f'Giá Cổ Phiếu {symbol}', 'MACD', 'RSI'),
                            row_heights=[0.5, 0.3, 0.2])
        
        # Stock price
        fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Giá Đóng Cửa', line=dict(color='blue')),
                      row=1, col=1)
        
        # Buy/Sell signals
        df['signals'] = signals
        buy_signals = df[df['signals'] == 'BUY']
        sell_signals = df[df['signals'] == 'SELL']
        fig.add_trace(go.Scatter(x=buy_signals['date'], y=buy_signals['close'], mode='markers', name='Tín Hiệu Mua',
                                 marker=dict(symbol='triangle-up', size=10, color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_signals['date'], y=sell_signals['close'], mode='markers', name='Tín Hiệu Bán',
                                 marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=df['date'], y=macd, mode='lines', name='Đường MACD', line=dict(color='blue')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=macdsignal, mode='lines', name='Đường Tín Hiệu', line=dict(color='red')),
                      row=2, col=1)
        fig.add_trace(go.Bar(x=df['date'], y=macdhist, name='Histogram', marker=dict(color='gray', opacity=0.5)),
                      row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df['date'], y=rsi, mode='lines', name='RSI', line=dict(color='purple')),
                      row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Quá Mua", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Quá Bán", row=3, col=1)
        
        # Layout
        fig.update_layout(
            title=f'Phân Tích {symbol} với MACD và RSI',
            yaxis_title='Giá (USD)',
            yaxis2_title='MACD',
            yaxis3_title='RSI',
            xaxis3_title='Ngày',
            height=1000,
            showlegend=True,
            template='plotly_white'
        )
        
        # Save plot
        plot_file = f"{symbol.lower()}_macd_rsi.html"
        fig.write_html(plot_file)
        
        # Return results
        return {
            "symbol": symbol.upper(),
            "last_price": round(prices[-1], 2),
            "last_macd": round(macd[-1], 2) if not np.isnan(macd[-1]) else None,
            "last_signal": round(macdsignal[-1], 2) if not np.isnan(macdsignal[-1]) else None,
            "last_histogram": round(macdhist[-1], 2) if not np.isnan(macdhist[-1]) else None,
            "last_rsi": round(rsi[-1], 2) if not np.isnan(rsi[-1]) else None,
            "last_signal_action": signals[-1],
            "plot_file": plot_file
        }
    except Exception as e:
        return {"error": f"Error calculating MACD and RSI for {symbol}: {str(e)}"}
    

# Guardrail: block crypto
class GuardrailOutput(BaseModel):
    allow: bool
    reason: str

crypto_keywords = {"bitcoin", "btc", "crypto", "doge", "solana", "eth", "ethereum"}

async def reject_invalid_tickers(ctx, agent, input_data):
    input_text = input_data.lower()
    if any(word in input_text for word in crypto_keywords):
        return GuardrailFunctionOutput(
            output_info=GuardrailOutput(allow=False, reason="Crypto queries are not supported."),
            tripwire_triggered=True
        )
    return GuardrailFunctionOutput(
        output_info=GuardrailOutput(allow=True, reason="OK"),
        tripwire_triggered=False
    )

# Compliance Agent
compliance_agent = Agent(
    name="Compliance Agent",
    instructions="You respond to questions requiring regulatory caution. Do not provide legal advice but offer general guidance on compliance-related topics."
)

compliance_handoff = handoff(
    agent=compliance_agent,
    input_filter=handoff_filters.remove_all_tools
)

# Trading Assistant Agent
trading_assistant_forecast = Agent(
    name="Trading Assistant forecast future price",
    instructions="""
You are a stock trading assistant focused on providing accurate and concise stock information. 
Your responsibilities include:

- Using provided tools to fetch and summarize stock data.
- Reporting results in clear markdown format, using tables for numerical data. 
- Using get_stock_forecast for future price predictions when requested.
- Model forecast: lstm, nbeats, transformer, tcn.
- If the user requests a forecast, they can specify the model by saying 'use [model]', 
- If not specified, the default is nbeats.

Rules and constraints:
- Always use tools for real-time or provided data; never guess or assume information.
- Do not include disclaimers or extra context unless asked.
""",
    tools=[get_stock_forecast],
    model="gpt-4o-mini"
)

# Trading Assistant Agent
trading_assistant = Agent(
    name="Trading Assistant",
    instructions="""
You are a stock trading assistant focused on providing accurate and concise stock information. Your responsibilities include:

- Using provided tools to fetch and summarize stock data.
- Reporting results in clear markdown format, using tables for numerical data.
- Providing BUY, SELL, or HOLD recommendations only when explicitly requested, based solely on retrieved data.
- Escalating legal or regulatory questions to the Compliance Agent.
- Refraining from answering non-investment-related queries.
RSI analysis:
- Use RSI_analysis to analyze the RSI of the stock.
- Report the RSI analysis in the to user the Evaluate RSI and give them recommendation with explaination.
MACD analysis:
- Using get_stock_macd_rsi tools to fetch and summarize stock data, including technical indicators (MACD, RSI), company information, and forecasts.
- Reporting results in clear markdown format, using tables for numerical data.
- Providing BUY, SELL, or HOLD recommendations only when explicitly requested, based solely on retrieved data.
- Analyzing MACD and RSI indicators to detect trading signals (e.g., MACD crossovers, RSI overbought/oversold).
- Generating interactive plots for MACD and RSI when requested, saved as HTML files.

Rules and constraints:
- Always use tools for real-time or provided data; never guess or assume information.
- Do not include disclaimers or extra context unless asked.
- If a tool fails or no data is returned, state this clearly without assumptions.
- Use markdown: bold headers, code formatting for tickers, tables for indicators.

Your role is to assist with data-driven stock trading decisions, not to provide general advice or opinions.
""",
    tools=[get_stock_company_info, get_stock_technical, RSI_analysis, get_stock_macd_rsi],
    input_guardrails=[InputGuardrail(guardrail_function=reject_invalid_tickers)],
    handoffs=[compliance_handoff, trading_assistant_forecast],
    handoff_description="Compliance_agent",
    model="gpt-4o-mini"
)

# Workflow Simulation
config = RunConfig(
    model_settings=ModelSettings(
        temperature=0.6
    )
)

async def main():
    output = []
    queries = ["What is the price of NVDA?",

               "What is RSI of NVDA?",
               "Analyze NVDA RSI and MACD and give me recommendation",
               ]
    for i, query in enumerate(queries):
        try:
            print(f'Q_{i+1}: {query}')
            result = await Runner.run(trading_assistant, query, run_config=config)
            print(f'A: {result.final_output}')  # Use final_output instead of result[0]['content']
            output.append(result)
        except Exception as e:
            if hasattr(e, 'guardrail_result'):
                reason = e.guardrail_result.output.output_info.reason
                print(f"Guardrail blocked input: {reason}")
            else:
                print(f"Error: {str(e)}")
            continue
        print("\n", 175 * "-", "\n")
    return queries, output

# Execute Workflow
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    if loop.is_running():
        queries, result_list = asyncio.run(main())
    else:
        queries, result_list = asyncio.run(main())