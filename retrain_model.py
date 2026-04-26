import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── 1. Download Data ─────────────────────────────────────
stock = yf.download('AAPL', start='2010-01-01', end='2025-01-01')
close_prices = stock[['Close']].values
print(f"✅ Total data points: {len(close_prices)}")

# ── 2. Scale Data ────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# ── 3. Create Sequences ──────────────────────────────────
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data  = scaled_data[train_size:]

X_train, y_train = create_sequences(train_data)
X_test,  y_test  = create_sequences(test_data)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

print(f"✅ X_train shape: {X_train.shape}")
print(f"✅ X_test  shape: {X_test.shape}")

# ── 4. Build LSTM Model ──────────────────────────────────
model = Sequential()

# Layer 1
model.add(LSTM(units=100, return_sequences=True,
               input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.1))

# Layer 2
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.1))

# Layer 3
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.1))

# Dense Layers
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=1))

# ── 5. Compile ───────────────────────────────────────────
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
print("\n🚀 Training started — this may take 20-40 mins...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ── 8. Evaluate ──────────────────────────────────────────
predictions   = model.predict(X_test)
predictions   = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(y_test_actual, predictions)
mae = mean_absolute_error(y_test_actual, predictions)

print(f"\n📊 Mean Squared Error  (MSE): {mse:.2f}")
print(f"📊 Mean Absolute Error (MAE): ${mae:.2f}")
print(f"📊 Epochs trained: {len(history.history['loss'])}")

# ── 9. Save Model ────────────────────────────────────────
model.save('stock_lstm_model.keras')
print("\n✅ Improved model saved as stock_lstm_model.keras")