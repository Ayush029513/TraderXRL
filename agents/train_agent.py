import sys
import os

sys.path.append(os.getcwd())

import pandas as pd
from stable_baselines3 import PPO
from env.trading_env import TradingEnv

# Load data
df = pd.read_csv("data/features.csv")

# Train/Test Split (80% Train, 20% Test)
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df = df.iloc[split_idx:].reset_index(drop=True)

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Create training environment
env = TradingEnv(train_df)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    batch_size=128,
    n_steps=4096,
    gamma=0.995,
    ent_coef=0.01
)

model.learn(
    total_timesteps=1000000
)

# Save model
model.save(
    "models/ppo_trading_agent"
)

print("Training Complete")
