import os
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

# === Flag: Set to True to retrain all models
RETRAIN_MODELS = True

# === Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# === Feature Engineering
def create_features(df):
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD5'] = df['Close'].rolling(window=5).std()
    df['Volume_Change'] = df['Volume'].pct_change()

    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    features = ['Return', 'MA5', 'MA10', 'MA20', 'STD5', 'Volume_Change', 'RSI']
    return df, features

# === Save/Load Functions
def save_model(model, name):
    with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)

def load_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_lstm_model(model, name):
    model.save(os.path.join(MODEL_DIR, f"{name}.h5"))

def load_lstm_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.h5")
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

# === LSTM Prep
def prepare_lstm_data(df, features, window_size=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i])
        y.append(df['Target'].values[i])
    return np.array(X), np.array(y), scaler

# === Model Training
def train_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model, accuracy_score(y_test, model.predict(X_test)), f1_score(y_test, model.predict(X_test))

def train_xgb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model, accuracy_score(y_test, model.predict(X_test)), f1_score(y_test, model.predict(X_test))

def train_lstm(X, y):
    X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
    y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test),
              callbacks=[EarlyStopping(patience=3)], verbose=0)
    preds = (model.predict(X_test) > 0.5).astype(int)
    return model, accuracy_score(y_test, preds), f1_score(y_test, preds)

# === Signal Predictions
def predict_signal(model, last_row):
    return "BUY" if model.predict(last_row)[0] == 1 else "SELL"

def predict_lstm(model, df, features, scaler, window_size=10):
    latest_data = scaler.transform(df[features].values[-window_size:])
    X_input = np.expand_dims(latest_data, axis=0)
    pred = model.predict(X_input)[0][0]
    return "BUY" if pred > 0.5 else "SELL"

# === Main Logic
def process_stock(symbol):
    print(f"\n📈 Processing {symbol}...")
    df = yf.download(symbol + ".NS", period="3y", interval="1d", progress=False)
    if df.empty or len(df) < 150:
        print("  ❌ Not enough data.")
        return

    df, features = create_features(df)
    X_tab = df[features]
    y = df['Target']

    # === RF
    retrain = RETRAIN_MODELS or not os.path.exists(os.path.join(MODEL_DIR, f"{symbol}_rf.pkl"))
    if retrain:
        rf_model, acc, f1 = train_rf(X_tab, y)
        save_model(rf_model, f"{symbol}_rf")
        print(f"  ✅ Trained RF | Accuracy: {acc:.2f} | F1: {f1:.2f}")
    else:
        rf_model = load_model(f"{symbol}_rf")
        print("  ✅ Loaded RF")

    # === XGB
    retrain = RETRAIN_MODELS or not os.path.exists(os.path.join(MODEL_DIR, f"{symbol}_xgb.pkl"))
    if retrain:
        xgb_model, acc, f1 = train_xgb(X_tab, y)
        save_model(xgb_model, f"{symbol}_xgb")
        print(f"  ✅ Trained XGB | Accuracy: {acc:.2f} | F1: {f1:.2f}")
    else:
        xgb_model = load_model(f"{symbol}_xgb")
        print("  ✅ Loaded XGB")

    # === LSTM
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_scaler.pkl")
    retrain = RETRAIN_MODELS or not os.path.exists(os.path.join(MODEL_DIR, f"{symbol}_lstm.h5")) or not os.path.exists(scaler_path)
    if retrain:
        X_lstm, y_lstm, scaler = prepare_lstm_data(df, features)
        lstm_model, acc, f1 = train_lstm(X_lstm, y_lstm)
        save_lstm_model(lstm_model, f"{symbol}_lstm")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  ✅ Trained LSTM | Accuracy: {acc:.2f} | F1: {f1:.2f}")
    else:
        lstm_model = load_lstm_model(f"{symbol}_lstm")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("  ✅ Loaded LSTM")

    # === Prediction
    rf_signal = predict_signal(rf_model, X_tab.iloc[[-1]])
    xgb_signal = predict_signal(xgb_model, X_tab.iloc[[-1]])
    lstm_signal = predict_lstm(lstm_model, df, features, scaler)

    print(f"  🔎 RF Signal: {rf_signal}")
    print(f"  🔎 XGB Signal: {xgb_signal}")
    print(f"  🔎 LSTM Signal: {lstm_signal}")

    final_signal = "BUY" if [rf_signal, xgb_signal, lstm_signal].count("BUY") >= 2 else "SELL"
    print(f"  🟢 Final Signal: {final_signal}")

# === Run on multiple stocks
stocks = ["RELIANCE", "TCS", "INFY"]
for stock in stocks:
    process_stock(stock)
