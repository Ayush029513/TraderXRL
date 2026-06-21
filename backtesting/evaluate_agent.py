import sys
import os

sys.path.append(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from env.trading_env import TradingEnv

# Load data
df = pd.read_csv("data/features.csv")

# Use only the test set (20%)
split_idx = int(len(df) * 0.8)

test_df = df.iloc[split_idx:].reset_index(drop=True)

# Create environment
env = TradingEnv(test_df)

# Load trained model
model = PPO.load("models/ppo_trading_agent")

obs, info = env.reset()

done = False

portfolio_values = []

while not done:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(action)

    if env.current_step % 50 == 0:
        print("Step:", env.current_step)

    done = terminated or truncated

    portfolio_values.append(
        info["net_worth"]
    )
    # after loop finishes

equity_df = pd.DataFrame(
    {"NetWorth": portfolio_values}
)

equity_df.to_csv(
    "results/equity_curve.csv",
    index=False
)
done = terminated or truncated


print("\n===== RESULTS =====")
print("Final Net Worth:", round(info["net_worth"], 2))
print("Final Balance:", round(info["balance"], 2))
print("Shares Held:", info["shares_held"])
print("Dataset loaded:", len(df))
print("Test rows:", len(test_df))
print("Environment created")
print("Model loaded")
print("Environment reset")

plt.figure(figsize=(12, 6))
plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Trading Steps")
plt.ylabel("Net Worth")
plt.grid(True)

plt.savefig("results/equity_curve.png")
plt.close()

