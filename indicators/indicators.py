import pandas as pd
import numpy as np
import ta

# Read Yahoo Finance CSV
df = pd.read_csv("data/aapl.csv")

# Remove extra Yahoo Finance rows
df = df.iloc[3:].copy()

# Rename columns
df.columns = [
    "Date",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume"
]

# Convert numeric columns
numeric_cols = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Remove invalid rows
df.dropna(inplace=True)

# =========================
# RSI
# =========================

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"]
).rsi()

# =========================
# MACD
# =========================

macd = ta.trend.MACD(df["Close"])

df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()

# =========================
# SMA
# =========================

df["SMA20"] = ta.trend.SMAIndicator(
    close=df["Close"],
    window=20
).sma_indicator()

df["SMA50"] = ta.trend.SMAIndicator(
    close=df["Close"],
    window=50
).sma_indicator()

# =========================
# ATR (Average True Range)
# =========================

df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
).average_true_range()

# =========================
# Daily Returns
# =========================

df["Returns"] = (
    df["Close"].pct_change()
)

# =========================
# Rolling Volatility
# =========================

df["Volatility"] = (
    df["Returns"]
    .rolling(20)
    .std()
)

# =========================
# Bollinger Bands
# =========================

bb = ta.volatility.BollingerBands(
    close=df["Close"],
    window=20,
    window_dev=2
)

df["BB_Upper"] = bb.bollinger_hband()
df["BB_Lower"] = bb.bollinger_lband()
df["BB_Middle"] = bb.bollinger_mavg()

# =========================
# Clean Data
# =========================

df.dropna(inplace=True)

# Save Features Dataset
df.to_csv(
    "data/features.csv",
    index=False
)

print(df.head())

print(
    f"\nFeatures dataset shape: {df.shape}"
)