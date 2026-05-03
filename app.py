from flask import Flask, render_template, request
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta, date
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
model = load_model('stock_lstm_model.keras')

@app.route('/', methods=['GET', 'POST'])
def index():
    error           = None
    predicted_price = None
    actual_price    = None
    symbol          = None
    input_date      = None
    tomorrow_date   = None
    direction       = None
    change          = None
    change_pct      = None
    graph_path      = None

    if request.method == 'POST':
        symbol     = request.form['symbol'].upper().strip()
        input_date = request.form['date'].strip()

        try:
            target_date   = datetime.strptime(input_date, '%Y-%m-%d')
            tomorrow_date = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
            start_date    = target_date - timedelta(days=100)

            stock = yf.download(symbol,
                                start=start_date.strftime('%Y-%m-%d'),
                                end=(target_date + timedelta(days=1)).strftime('%Y-%m-%d'))

            if stock.empty:
                error = f"No data found for '{symbol}'. Please check the symbol."
            elif len(stock) < 60:
                error = f"Not enough data before {input_date}. Try an earlier date."
            else:
                close_prices = stock[['Close']].values
                scaler       = MinMaxScaler(feature_range=(0, 1))
                scaled_data  = scaler.fit_transform(close_prices)

                last_60          = scaled_data[-60:].reshape(1, 60, 1)
                predicted_scaled = model.predict(last_60)
                predicted_price  = round(float(scaler.inverse_transform(predicted_scaled)[0][0]), 2)
                actual_price     = round(float(close_prices[-1][0]), 2)
                change           = round(predicted_price - actual_price, 2)
                change_pct       = round((change / actual_price) * 100, 2)
                direction        = "UP" if predicted_price > actual_price else "DOWN"

                # Generate 60-day graph
                last_60_prices = close_prices[-60:]
                dates_60       = stock.index[-60:]

                graph_folder = os.path.join(app.root_path, 'static', 'graphs')
                os.makedirs(graph_folder, exist_ok=True)
                graph_file = os.path.join(graph_folder, f'{symbol}_60days.png')

                fig, ax = plt.subplots(figsize=(12, 5))
                fig.patch.set_facecolor('#0d1120')
                ax.set_facecolor('#0d1120')
                ax.plot(dates_60, last_60_prices,
                        color='#00d4ff', linewidth=2, label='Close Price')
                ax.fill_between(dates_60,
                                last_60_prices.flatten(),
                                alpha=0.15, color='#00d4ff')
                ax.scatter(dates_60[-1], actual_price,
                           color='#00ff88', s=80, zorder=5,
                           label=f'Today: ${actual_price}')
                ax.scatter(dates_60[-1], predicted_price,
                           color='#ff9900', s=80, zorder=5,
                           marker='*', label=f'Predicted: ${predicted_price}')
                ax.set_title(f'{symbol} — Last 60 Days Closing Price',
                             color='#a0b4cc', fontsize=14, pad=15)
                ax.set_xlabel('Date', color='#556a80', fontsize=10)
                ax.set_ylabel('Price (USD $)', color='#556a80', fontsize=10)
                ax.tick_params(colors='#556a80')
                ax.spines['bottom'].set_color('#1e2a3a')
                ax.spines['top'].set_color('#1e2a3a')
                ax.spines['left'].set_color('#1e2a3a')
                ax.spines['right'].set_color('#1e2a3a')
                ax.grid(True, color='#1e2a3a', linewidth=0.8)
                ax.legend(facecolor='#0d1120', labelcolor='#a0b4cc',
                          fontsize=9, loc='upper left')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(graph_file, dpi=120,
                            facecolor='#0d1120', bbox_inches='tight')
                plt.close()

                graph_path = f'graphs/{symbol}_60days.png'

        except ValueError:
            error = "Invalid date format. Please use YYYY-MM-DD"
        except Exception as e:
            error = f"Something went wrong: {str(e)}"

    today = date.today().strftime('%Y-%m-%d')

    return render_template('index.html',
                           symbol=symbol,
                           input_date=input_date,
                           tomorrow_date=tomorrow_date,
                           predicted_price=predicted_price,
                           actual_price=actual_price,
                           direction=direction,
                           change=change,
                           change_pct=change_pct,
                           graph_path=graph_path,
                           error=error,
                           today=today)

if __name__ == '__main__':
    app.run(debug=False)