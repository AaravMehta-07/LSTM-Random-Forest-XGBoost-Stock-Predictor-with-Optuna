# LSTM-Random-Forest-XGBoost-Stock-Predictor-with-Optuna
A hybrid AI-based stock market prediction system using LSTM, Random Forest, and XGBoost, built for real-world deployment with Optuna-powered tuning, feature-rich engineering, and ensemble prediction logic. Designed to optimize F1 score and accuracy, this system aims to generate reliable buy/sell signals on stocks.
# 📈 LSTM + Random Forest + XGBoost Stock Predictor

---

## 🚀 About the Project
This project integrates:
- 🔁 **Recurrent Neural Networks (LSTM)** for sequential financial patterns
- 🌲 **Random Forest** for ensemble-based classification
- ⚡ **XGBoost** for gradient boosting decision trees
- 🎯 **Optuna** for automatic hyperparameter tuning (optional mode)
- 📊 **Backtesting Module** to simulate trading performance

> ⚙️ Built by a Computer Engineering student to demonstrate real-world ML/AI skills in finance and time series prediction.

---

## 📌 Features

- ✔️ Ensemble of 3 models: LSTM + RF + XGBoost
- ✔️ Flag-based retraining (no need to retrain every time)
- ✔️ Real stock data from Yahoo Finance
- ✔️ Feature-rich engineering: RSI, Moving Averages, Volatility, Volume
- ✔️ Backtesting for historical performance validation
- ✔️ Soft voting for final trade signal (BUY / SELL)
- ✔️ CLI-based output, no GUI bloat
- ✔️ Saved model reuse (`models/` folder)

---

## 🧠 Technologies Used

| Category         | Stack                                |
|------------------|---------------------------------------|
| Language         | Python 3.10                           |
| ML Models        | RandomForestClassifier, XGBClassifier |
| Deep Learning    | TensorFlow / Keras LSTM               |
| Optimization     | Optuna                                |
| Data Source      | yfinance                              |
| Indicators       | RSI, MA5/10/20, Volatility, Volume    |

---

## 📂 Project Structure
LSTM-RandomForest-XGBoost-Stock-Predictor/
├── LSTM+Random Forest+XGboost Stock Predictor.py
├── LSTM+Random Forest+XGboost Stock Predictor with optuna.py
├── backtest.py
├── models/ # Contains saved models
├── README.md
├── LICENSE # MIT License
