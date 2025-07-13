# FILE: agent_gemini.py

# Basic libraries
import time
import json
import requests
import warnings
import os
import asyncio
import configparser

# data analysis
import numpy as np
import pandas as pd

# financial libs
import talib # terminal install -> pip install TA-Lib

# Gemini libs
import google.generativeai as genai

try:
    from darts import TimeSeries
    from darts.models import RNNModel, NBEATSModel, TransformerModel, TCNModel
    from darts.dataprocessing.transformers import Scaler
    from darts.metrics import mape
except ImportError as e:
    print(f"Darts library not found. Forecasting tool will not work. Please install: pip install darts")
    TimeSeries = None # Set to None to handle gracefully

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gradio as gr

warnings.filterwarnings('ignore')


try:
    key_file_name = 'apikey'  # Sử dụng tên file thực tế
    path = os.getcwd()  # Sử dụng thư mục hiện tại
    config = configparser.ConfigParser()
    config.read(os.path.join(path, key_file_name))
    
    # Lấy key của Google Gemini
    google_api_key = config['google']['api_key']
    genai.configure(api_key=google_api_key)
    
    # Key cho Financial Modeling Prep vẫn cần thiết
    fmp_key = config['financialmodelingprep']['api_key']
except Exception as e:
    print(f"Lỗi khi đọc file API key: {e}")
    print("Hãy đảm bảo file text 'apikey' tồn tại và có section [google] và [financialmodelingprep]")
    exit()

def get_historical_data(symbol: str) -> pd.DataFrame:
    """
    Fetch historical closing prices from FMP API.
    Returns DataFrame with 'date' and 'close' columns, ensuring daily frequency.
    """
    start_date = "2024-06-01"
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
        df = df.rename(columns={'index': 'date'}, inplace=False)
        # Lưu dữ liệu vào file csv
        df.to_csv(f'{symbol}_historical_data.csv', index=False)
        return df
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

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

def get_stock_technical(symbol: str) -> dict:
    """
    Calculate technical indicators (SMA, RSI, MACD) from FMP API historical data.
    """
    try:
        df = get_historical_data(symbol)
        if df.empty: return {"error": f"No historical data for symbol: {symbol}"}
        close_np = df["close"].to_numpy(dtype='float64')
        if len(close_np) < 200: return {"error": f"Not enough data to compute indicators for {symbol} (need at least 200 days)"}
        sma_50, sma_200 = talib.SMA(close_np, 50), talib.SMA(close_np, 200)
        rsi = talib.RSI(close_np, 14)
        macd, macdsignal, _ = talib.MACD(close_np)
        return {
            "symbol": symbol.upper(), "last_price": round(close_np[-1], 2),
            "SMA_50d": round(sma_50[-1], 2), "SMA_200d": round(sma_200[-1], 2),
            "RSI": round(rsi[-1], 2), "MACD": round(macd[-1], 2),
            "MACD_signal": round(macdsignal[-1], 2)
        }
    except Exception as e: return {"error": f"Unhandled error for symbol {symbol}: {str(e)}"}

def get_model_path(symbol: str, model_type: str) -> str:
    """Generate a unique path for saving/loading model weights."""
    model_dir = os.path.join(os.getcwd(), "saved_models_NBEATS")
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    return os.path.join(model_dir, f"{model_type}_{symbol.lower()}.pt")

def get_stock_forecast(symbol: str, horizon: int = 5, model_type: str = 'nbeats') -> dict:
    """
    Forecast future stock prices using Darts models (nbeats, lstm, transformer, tcn).
    Defaults to nbeats if model_type is not specified.
    """
    if TimeSeries is None: return {"error": "Darts library is not installed."}
    try:
        df = get_historical_data(symbol)
        if df.empty: return {"error": f"No historical data for symbol: {symbol}"}
        
        series = TimeSeries.from_dataframe(df, 'date', 'close', fill_missing_dates=True, freq='D')
        train, val = series.split_after(0.8)
        
        input_chunk_length, model_path = 30, get_model_path(symbol, model_type)
        model = None
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            if model_type == 'nbeats': model = NBEATSModel.load(model_path)
            # Add other model loading logic here if needed
        
        if model is None:
            print(f"Training new {model_type} model for {symbol}...")
            scaler = Scaler()
            train_scaled = scaler.fit_transform(train)
            val_scaled = scaler.transform(val)
            
            if model_type == 'nbeats':
                model = NBEATSModel(input_chunk_length=input_chunk_length, output_chunk_length=horizon, n_epochs=50, random_state=42)
            # Add other model creation logic here
            else: return {"error": f"Model type '{model_type}' not fully supported in this version. Use 'nbeats'."}
            
            model.fit(train_scaled, val_series=val_scaled, verbose=False)
            model.save(model_path)
            print(f"Model saved to {model_path}")

        # Forecast on the full series (scaled) and inverse transform
        scaler = Scaler()
        series_scaled = scaler.fit_transform(series)
        forecast_scaled = model.predict(n=horizon, series=series_scaled)
        forecast = scaler.inverse_transform(forecast_scaled)

        return {"symbol": symbol.upper(), "forecast_prices": [round(p, 2) for p in forecast.values().flatten().tolist()]}
    except Exception as e: return {"error": f"Error forecasting for symbol {symbol}: {str(e)}"}

def RSI_analysis(symbol: str, days: int = 14) -> dict:
    """
    Analyzes RSI to provide BUY/SELL/HOLD signals.
    """
    df = get_historical_data(symbol)
    if df.empty: return {"error": f"No historical data for symbol: {symbol}"}
    close_np = df["close"].to_numpy(dtype='float64')
    rsi = talib.RSI(close_np, timeperiod=days)
    current_rsi = rsi[-1]
    signals = "Neutral - HOLD"
    if current_rsi > 70: signals = "Overbought - Potential SELL"
    elif current_rsi < 30: signals = "Oversold - Potential BUY"
    return {"symbol": symbol.upper(), "RSI": round(current_rsi, 2), "signal": signals}

def get_stock_macd_rsi_with_plot(symbol: str) -> dict:
    """
    Calculate MACD and RSI, detect signals, and generate an interactive plot.
    Returns analysis and the path to the HTML plot file.
    """
    try:
        df = get_historical_data(symbol)
        if df.empty: return {"error": f"No historical data for {symbol}"}
        
        close = df['close'].astype('float64')
        macd, macdsignal, macdhist = talib.MACD(close)
        rsi = talib.RSI(close)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(f'Giá Cổ Phiếu & MACD {symbol}', 'RSI'))
        
        fig.add_trace(go.Scatter(x=df['date'], y=df['close'], name='Giá'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=macd, name='MACD'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=macdsignal, name='Signal Line'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=rsi, name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row="2", col="1")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row="2", col="1")
        
        fig.update_layout(title=f'Phân Tích Kỹ Thuật {symbol}', height=800)
        
        plot_dir = "plots"
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        plot_file = os.path.join(plot_dir, f"{symbol.lower()}_plot.html")
        fig.write_html(plot_file)
        
        return {
            "symbol": symbol.upper(), "last_price": round(close.iloc[-1], 2),
            "last_rsi": round(rsi.iloc[-1], 2), "last_macd": round(macd.iloc[-1], 2),
            "plot_file": os.path.abspath(plot_file)
        }
    except Exception as e: return {"error": f"Error generating plot for {symbol}: {str(e)}"}

crypto_keywords = {"bitcoin", "btc", "crypto", "doge", "solana", "eth", "ethereum"}

def check_guardrail(query: str) -> bool:
    """Kiểm tra xem câu hỏi có chứa từ khóa bị cấm không."""
    if any(word in query.lower() for word in crypto_keywords):
        print("Guardrail Activated: Crypto queries are not supported.")
        return True
    return False

def run_gemini_agent(query: str):
    # 3.1. Xác định Agent và các tool
    tools = [
        get_stock_company_info,
        get_stock_technical,
        get_stock_forecast,
        RSI_analysis,
        get_stock_macd_rsi_with_plot,
    ]

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="""You are a helpful financial assistant. 
        - Use the provided tools to answer user questions about stocks.
        - Analyze the results from the tools to provide a summary.
        - If a user asks for a plot or chart, use the 'get_stock_macd_rsi_with_plot' tool.
        - If a user asks for a forecast or prediction, use the 'get_stock_forecast' tool.
        - For general technicals, use 'get_stock_technical'.
        - For company info, use 'get_stock_company_info'.
        - For a simple RSI signal, use 'RSI_analysis'.
        - Present data in clear markdown tables.
        - Never make up information. If a tool fails or returns an error, report that error to the user.
        - Respond in Vietnamese.
        """
    )
    
    # Bắt đầu phiên trò chuyện với Gemini
    chat = model.start_chat()
    
    print(f"Sending to Gemini: '{query}'")
    # Gửi yêu cầu đầu tiên của người dùng
    response = chat.send_message(query)

    # 3.2. Vòng lặp xử lý Tool Calling
    while response.candidates[0].content.parts[0].function_call.name:
        function_call = response.candidates[0].content.parts[0].function_call
        
        # Lấy tên hàm và đối số mà model muốn gọi
        tool_name = function_call.name
        tool_args = {key: value for key, value in function_call.args.items()}
        
        print(f"Tool Call: {tool_name}({tool_args})")
        
        # Tìm sử dụng tool
        tool_function = next((t for t in tools if t.__name__ == tool_name), None)
        
        if tool_function:
            try:
                tool_output = tool_function(**tool_args)
                # Gửi kết quả của tool 
                response = chat.send_message(
                    genai.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={'output': json.dumps(tool_output)}
                        )
                    )
                )
            except Exception as e:
                print(f"Error executing tool {tool_name}: {e}")
                response = chat.send_message(
                     genai.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={'error': str(e)}
                        )
                    )
                )
        else:
            print(f"Error: Tool '{tool_name}' not found.")
            break
            
    # 3.3. In ra kết quả cuối cùng
    return response.text

chat_memory = []  # Chat memory

def main():
    # Danh sách các câu hỏi cũ bây giờ được dùng làm ví dụ
    example_queries = [
        "Giá cổ phiếu NVDA là bao nhiêu, và thông tin công ty là gì?",
        "Phân tích chỉ số RSI của cổ phiếu Apple.",
        "Tôi muốn xem biểu đồ phân tích kỹ thuật của MSFT.",
        "Dự báo giá cổ phiếu Tesla trong 5 ngày tới.",
        "Giá bitcoin hôm nay thế nào?" # Ví dụ này sẽ bị Guardrail chặn
    ]

    # Sử dụng theme 'Soft' của Gradio để có giao diện hiện đại
    with gr.Blocks(theme=gr.themes.Soft(), title="Trợ lý AI Tài chính") as demo:
        gr.Markdown(
            """
            # Financial AI Agent
            Chào mừng bạn! Hãy hỏi tôi về cổ phiếu, phân tích kỹ thuật, hoặc tin tức thị trường.
            """
        )
        
        chatbot = gr.Chatbot(label="Hộp thoại", height=500)
        
        with gr.Row():
            msg = gr.Textbox(
                label="Nhập câu hỏi của bạn",
                placeholder="Ví dụ: Phân tích cổ phiếu VNM...",
                scale=4, # Làm cho ô nhập liệu rộng hơn
            )
            send_btn = gr.Button("Gửi", variant="primary", scale=1)

        gr.Examples(
            examples=example_queries,
            inputs=msg,
            label="Hoặc chọn một câu hỏi ví dụ"
        )

        def respond(message, chat_history):
            """
            Hàm xử lý khi người dùng gửi tin nhắn.
            """
            # 1. Kiểm tra Guardrail trước khi xử lý
            if check_guardrail(message):
                reply = "Rất tiếc, tôi không thể trả lời các câu hỏi liên quan đến tiền điện tử. Vui lòng hỏi về các chủ đề tài chính khác như cổ phiếu."
                chat_history.append((message, reply))
                return "", chat_history

            # 2. Nếu không bị chặn, gọi agent để lấy câu trả lời
            try:
                reply = run_gemini_agent(message)
            except Exception as e:
                print(f"ERROR: Đã xảy ra lỗi: {e}")
                reply = f"Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại. Lỗi: {e}"
            
            # 3. Cập nhật lịch sử chat
            chat_memory.append((message, reply))  # Cập nhật bộ nhớ để lưu
            chat_history.append((message, reply)) # Cập nhật giao diện
            
            # Xóa nội dung trong ô nhập liệu và cập nhật chatbot
            return "", chat_history

        send_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        
        with gr.Row():
            save_btn = gr.Button(" Lưu đoạn hội thoại")
            status_text = gr.Textbox(label="Trạng thái", interactive=False, placeholder="Trạng thái lưu...")

        def save_chat():
            if not chat_memory:
                return "Chưa có nội dung hội thoại để lưu."
            try:
                with open("chatlog.txt", "w", encoding="utf-8") as f:
                    f.write("=== LỊCH SỬ HỘI THOẠI ===\n\n")
                    for user_msg, bot_msg in chat_memory:
                        f.write(f"Người dùng: {user_msg}\n")
                        f.write(f" Bot: {bot_msg}\n")
                        f.write("-" * 30 + "\n")
                return " Đã lưu thành công vào file chatlog.txt"
            except Exception as e:
                return f" Lỗi khi lưu file: {e}"

        save_btn.click(save_chat, inputs=[], outputs=[status_text])

    # Chạy ứng dụng
    demo.launch()

# Điểm khởi đầu của chương trình
if __name__ == "__main__":
    main()