import pandas as pd
import numpy as np

equity = pd.read_csv("results/equity_curve.csv")

returns = equity["NetWorth"].pct_change().dropna()

if returns.std() > 0:
    sharpe = np.sqrt(252) * (
        returns.mean() / returns.std()
    )
else:
    sharpe = 0

rolling_max = equity["NetWorth"].cummax()

drawdown = (
    equity["NetWorth"] - rolling_max
) / rolling_max

max_drawdown = drawdown.min()

print("\n===== RISK REPORT =====")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")