import pandas as pd

df = pd.read_csv("data/features.csv")

initial_balance = 10000

first_price = df["Close"].iloc[0]
last_price = df["Close"].iloc[-1]

shares = initial_balance / first_price

buy_hold_value = shares * last_price

buy_hold_return = (
    (buy_hold_value - initial_balance)
    / initial_balance
) * 100

agent_value = 29919.71

agent_return = (
    (agent_value - initial_balance)
    / initial_balance
) * 100

print("\n===== PERFORMANCE REPORT =====")

print(
    f"Buy & Hold Value: ${buy_hold_value:.2f}"
)

print(
    f"Buy & Hold Return: {buy_hold_return:.2f}%"
)

print(
    f"Agent Value: ${agent_value:.2f}"
)

print(
    f"Agent Return: {agent_return:.2f}%"
)

print(
    f"Alpha: {(agent_return - buy_hold_return):.2f}%"
)