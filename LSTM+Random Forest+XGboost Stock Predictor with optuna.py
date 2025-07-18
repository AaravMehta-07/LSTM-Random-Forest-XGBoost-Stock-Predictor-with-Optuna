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
import optuna
import warnings

warnings.filterwarnings("ignore")

RETRAIN_MODELS = True

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

def prepare_lstm_data(df, features, window_size=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i])
        y.append(df['Target'].values[i])
    return np.array(X), np.array(y), scaler

# === Optuna RF
def train_rf_optuna(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
        }
        model = RandomForestClassifier(**params, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10, show_progress_bar=False)
    best_params = study.best_params
    final_model = RandomForestClassifier(**best_params, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)
    return final_model, accuracy_score(y_test, preds), f1_score(y_test, preds)

# === Optuna XGB
def train_xgb_optuna(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10, show_progress_bar=False)
    best_params = study.best_params
    final_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)
    return final_model, accuracy_score(y_test, preds), f1_score(y_test, preds)

# === Optuna LSTM
def train_lstm_optuna(X, y):
    def objective(trial):
        units_1 = trial.suggest_int("units_1", 32, 128)
        units_2 = trial.suggest_int("units_2", 16, 64)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        epochs = trial.suggest_int("epochs", 10, 30)

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential([
            LSTM(units_1, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(dropout),
            LSTM(units_2),
            Dropout(dropout),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_test, y_test), verbose=0,
                  callbacks=[EarlyStopping(patience=3)])
        preds = (model.predict(X_test) > 0.5).astype(int)
        return f1_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5, show_progress_bar=False)
    best = study.best_params

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(best["units_1"], return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(best["dropout"]),
        LSTM(best["units_2"]),
        Dropout(best["dropout"]),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=best["epochs"], batch_size=best["batch_size"],
              validation_data=(X_test, y_test), verbose=0,
              callbacks=[EarlyStopping(patience=3)])
    preds = (model.predict(X_test) > 0.5).astype(int)
    return model, accuracy_score(y_test, preds), f1_score(y_test, preds)

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
    rf_model, acc, f1 = train_rf_optuna(X_tab, y)
    save_model(rf_model, f"{symbol}_rf")
    print(f"  ✅ Trained RF | Accuracy: {acc:.2f} | F1: {f1:.2f}")

    # === XGB
    xgb_model, acc, f1 = train_xgb_optuna(X_tab, y)
    save_model(xgb_model, f"{symbol}_xgb")
    print(f"  ✅ Trained XGB | Accuracy: {acc:.2f} | F1: {f1:.2f}")

    # === LSTM
    X_lstm, y_lstm, scaler = prepare_lstm_data(df, features)
    lstm_model, acc, f1 = train_lstm_optuna(X_lstm, y_lstm)
    save_lstm_model(lstm_model, f"{symbol}_lstm")
    with open(os.path.join(MODEL_DIR, f"{symbol}_lstm_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✅ Trained LSTM | Accuracy: {acc:.2f} | F1: {f1:.2f}")

    # === Predictions
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
