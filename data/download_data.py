import yfinance as yf

ticker = "AAPL"

data = yf.download(
    ticker,
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=True
)

data.to_csv("data/aapl.csv")

print(data.head())
print(f"Downloaded {len(data)} rows")