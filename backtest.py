import os
import pickle
import numpy as np
import yfinance as yf
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load functions
def load_model_pickle(name):
    with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)

def load_scaler(symbol):
    with open(os.path.join(MODEL_DIR, f"{symbol}_lstm_scaler.pkl"), "rb") as f:
        return pickle.load(f)

def create_features(df):
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD5'] = df['Close'].rolling(window=5).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    features = ['Return', 'MA5', 'MA10', 'MA20', 'STD5', 'Volume_Change', 'RSI']
    return df, features

def prepare_lstm_data(df, features, window_size=10):
    scaler = load_scaler(symbol)
    scaled = scaler.transform(df[features])
    return scaled, scaler

def backtest(symbol):
    print(f"\n🔁 Backtesting {symbol}...")

    # Load models
    rf = load_model_pickle(f"{symbol}_rf")
    xgb = load_model_pickle(f"{symbol}_xgb")
    lstm = load_model(os.path.join(MODEL_DIR, f"{symbol}_lstm.h5"))
    scaler = load_scaler(symbol)

    df = yf.download(f"{symbol}.NS", period="3y", interval="1d", progress=False)
    df, features = create_features(df)

    initial_cash = 100000
    cash = initial_cash
    holdings = 0
    equity_curve = []
    window_size = 10

    for i in range(window_size, len(df) - 1):
        X_tab = df[features].iloc[[i]]
        X_lstm_seq = scaler.transform(df[features].iloc[i - window_size + 1:i + 1])
        X_lstm_seq = np.expand_dims(X_lstm_seq, axis=0)

        rf_pred = rf.predict(X_tab)[0]
        xgb_pred = xgb.predict(X_tab)[0]
        lstm_pred = int(lstm.predict(X_lstm_seq)[0][0] > 0.5)

        final_signal = 1 if [rf_pred, xgb_pred, lstm_pred].count(1) >= 2 else 0

        price_today = df.iloc[i]['Close']
        price_tomorrow = df.iloc[i + 1]['Close']

        if final_signal == 1 and cash > 0:
            holdings = cash / price_today
            cash = 0
        elif final_signal == 0 and holdings > 0:
            cash = holdings * price_today
            holdings = 0

        total = cash + holdings * price_today
        equity_curve.append(total)

    final_value = cash + holdings * df.iloc[-1]['Close']
    total_return = (final_value - initial_cash) / initial_cash * 100
    print(f"💰 Final Value: ₹{final_value:.2f}")
    print(f"📈 Total Return: {total_return:.2f}%")

    plt.plot(equity_curve)
    plt.title(f"{symbol} Strategy Equity Curve")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value (₹)")
    plt.grid()
    plt.show()


# === Run backtest for your stocks
for symbol in ["RELIANCE", "TCS", "INFY"]:
    backtest(symbol)
