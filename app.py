import gradio as gr
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Load model
model = load_model('stock_lstm_model.keras')

def predict_stock(symbol, date_str):
    try:
        symbol      = symbol.upper().strip()
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        tomorrow    = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
        start_date  = target_date - timedelta(days=100)

        # Fetch data
        stock = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=(target_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            progress=False
        )

        if stock.empty:
            return f"❌ Symbol '{symbol}' not found. Please check!", None

        if len(stock) < 60:
            return "❌ Not enough data. Try an earlier date!", None

        close_prices = stock[['Close']].values
        scaler       = MinMaxScaler(feature_range=(0, 1))
        scaled_data  = scaler.fit_transform(close_prices)

        # Predict
        last_60          = scaled_data[-60:].reshape(1, 60, 1)
        predicted_scaled = model.predict(last_60)
        predicted_price  = round(float(scaler.inverse_transform(predicted_scaled)[0][0]), 2)
        actual_price     = round(float(close_prices[-1][0]), 2)
        change           = round(predicted_price - actual_price, 2)
        change_pct       = round((change / actual_price) * 100, 2)
        direction        = "📈 UP" if predicted_price > actual_price else "📉 DOWN"

        # Generate graph
        last_60_prices = close_prices[-60:]
        dates_60       = stock.index[-60:]

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#0d1120')
        ax.set_facecolor('#0d1120')
        ax.plot(dates_60, last_60_prices,
                color='#00d4ff', linewidth=2, label='Close Price')
        ax.fill_between(dates_60, last_60_prices.flatten(),
                        alpha=0.15, color='#00d4ff')
        ax.scatter(dates_60[-1], actual_price,
                   color='#00ff88', s=100, zorder=5,
                   label=f'Today: ${actual_price}')
        ax.scatter(dates_60[-1], predicted_price,
                   color='#ff9900', s=100, zorder=5,
                   marker='*', label=f'Predicted: ${predicted_price}')
        ax.set_title(f'{symbol} — Last 60 Days Price',
                     color='#a0b4cc', fontsize=14, pad=15)
        ax.set_xlabel('Date', color='#556a80')
        ax.set_ylabel('Price (USD $)', color='#556a80')
        ax.tick_params(colors='#556a80')
        for spine in ax.spines.values():
            spine.set_color('#1e2a3a')
        ax.grid(True, color='#1e2a3a', linewidth=0.8)
        ax.legend(facecolor='#0d1120', labelcolor='#a0b4cc', fontsize=9)
        plt.xticks(rotation=45)
        plt.tight_layout()

        result = f"""
## 📊 Prediction Results for {symbol}

| Detail | Value |
|--------|-------|
| 📅 Today's Date | {date_str} |
| 🔮 Prediction Date | {tomorrow} |
| 💵 Today's Actual Price | **${actual_price}** |
| 🎯 Predicted Tomorrow's Price | **${predicted_price}** |
| 📊 Expected Change | **${change} ({change_pct}%)** |
| 📈 Direction | **{direction}** |

---
⚠️ *For educational purposes only. Not financial advice.*
        """

        return result, fig

    except Exception as e:
        return f"❌ Error: {str(e)}", None

# ── Gradio UI ────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="StockAI — Price Predictor",
    css="""
        .gradio-container { max-width: 1200px !important; }
        h1 { text-align: center; }
    """
) as demo:

    gr.Markdown("""
    # 📈 StockAI — AI Stock Price Predictor
    ### Powered by LSTM Neural Network trained on 20 companies
    Enter a stock symbol and today's date to predict **tomorrow's closing price!**
    """)

    with gr.Row():

        # Left Column — Inputs
        with gr.Column(scale=1):
            symbol_input = gr.Textbox(
                label="📌 Stock Symbol",
                placeholder="e.g. AAPL, TSLA, TCS.NS",
                value="AAPL"
            )
            date_input = gr.Textbox(
                label="📅 Today's Date (YYYY-MM-DD)",
                placeholder="e.g. 2024-06-15",
                value=datetime.today().strftime('%Y-%m-%d')
            )
            predict_btn = gr.Button(
                "🔮 Predict Tomorrow's Price",
                variant="primary",
                size="lg"
            )

            gr.Markdown("""
            ---
            ### 🔥 Popular Stocks
            **🇺🇸 US Stocks:**
            `AAPL` `TSLA` `GOOGL` `MSFT` `NVDA` `META` `AMZN`

            **🇮🇳 Indian Stocks:**
            `TCS.NS` `RELIANCE.NS` `INFY.NS` `HDFCBANK.NS` `WIPRO.NS`

            ---
            ### 🧠 Model Info
            - Trained on **20 companies**
            - **10 years** of data (2015-2025)
            - **4 LSTM layers**
            - **60 day** sequence length
            - MSE: **0.000169**
            """)

        # Right Column — Results
        with gr.Column(scale=2):
            result_output = gr.Markdown(
                value="#### Results will appear here after prediction! 👆"
            )
            graph_output = gr.Plot(label="📊 60 Day Price Chart")

    # Button click
    predict_btn.click(
        fn=predict_stock,
        inputs=[symbol_input, date_input],
        outputs=[result_output, graph_output]
    )

demo.launch()