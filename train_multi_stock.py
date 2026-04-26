import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# ── 1. Define Multiple Stocks ────────────────────────────
STOCKS = [
    # US Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'NVDA', 'META', 'NFLX', 'AMD',  'INTC',
    # US Finance & Others
    'JPM', 'GS', 'V', 'KO', 'NKE',
    # Indian Stocks
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS',
    'HDFCBANK.NS', 'WIPRO.NS'
]

SEQ_LEN    = 60
START_DATE = '2015-01-01'
END_DATE   = '2025-01-01'

# ── 2. Download & Prepare All Stock Data ─────────────────
all_X = []
all_y = []

print(f"📥 Downloading data for {len(STOCKS)} stocks...\n")

for symbol in STOCKS:
    try:
        print(f"  ⬇️  Fetching {symbol}...")
        stock = yf.download(symbol,
                            start=START_DATE,
                            end=END_DATE,
                            progress=False)

        if stock.empty or len(stock) < SEQ_LEN + 10:
            print(f"  ⚠️  Skipping {symbol} — not enough data")
            continue

        close_prices = stock[['Close']].values

        # Scale each stock independently
        scaler      = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Create sequences
        for i in range(SEQ_LEN, len(scaled_data)):
            all_X.append(scaled_data[i-SEQ_LEN:i, 0])
            all_y.append(scaled_data[i, 0])

        print(f"  ✅ {symbol} — {len(stock)} days added")

    except Exception as e:
        print(f"  ❌ {symbol} failed: {e}")

print(f"\n✅ Total sequences collected: {len(all_X)}")

# ── 3. Convert to Arrays ─────────────────────────────────
all_X = np.array(all_X)
all_y = np.array(all_y)

# Shuffle data so model doesn't overfit one stock
indices = np.random.permutation(len(all_X))
all_X   = all_X[indices]
all_y   = all_y[indices]

# Reshape for LSTM
all_X = all_X.reshape((all_X.shape[0], all_X.shape[1], 1))

# ── 4. Train / Test Split ────────────────────────────────
split     = int(len(all_X) * 0.8)
X_train   = all_X[:split]
y_train   = all_y[:split]
X_test    = all_X[split:]
y_test    = all_y[split:]

print(f"✅ Training samples : {len(X_train)}")
print(f"✅ Testing  samples : {len(X_test)}")

# ── 5. Build Model ───────────────────────────────────────
model = Sequential()

# Layer 1
model.add(LSTM(units=128, return_sequences=True,
               input_shape=(SEQ_LEN, 1)))
model.add(Dropout(0.1))

# Layer 2
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.1))

# Layer 3
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.1))

# Dense Layers
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ── 6. Callbacks ─────────────────────────────────────────
early_stop = EarlyStopping(monitor='val_loss',
                           patience=20,
                           restore_best_weights=True,
                           verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.3,
                              patience=8,
                              verbose=1)

# ── 7. Train ─────────────────────────────────────────────
print("\n🚀 Training on multi-stock data — this may take 30-60 mins...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ── 8. Evaluate ──────────────────────────────────────────
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"\n📊 Mean Squared Error  (MSE): {mse:.6f}")
print(f"📊 Mean Absolute Error (MAE): {mae:.6f}")
print(f"📊 Epochs trained: {len(history.history['loss'])}")

# ── 9. Save Model ─────────────────────────────────────────
model.save('stock_lstm_model.keras')
print("\n✅ Multi-stock model saved as stock_lstm_model.keras")