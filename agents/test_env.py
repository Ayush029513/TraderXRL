import sys
import os

sys.path.append(os.getcwd())

import pandas as pd
from env.trading_env import TradingEnv
# Load data
df = pd.read_csv("data/features.csv")

# Create environment
env = TradingEnv(df)

obs, info = env.reset()

print("Initial Observation:")
print(obs)

for i in range(10):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep {i+1}")
    print("Action:", action)
    print("Reward:", reward)
    print("Net Worth:", info["net_worth"])

    if terminated:
        break

print("\nEnvironment test completed.")