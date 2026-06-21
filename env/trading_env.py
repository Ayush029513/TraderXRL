import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):

    def __init__(self, df, initial_balance=10000):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance

        # Actions
        # 0 = Hold
        # 1 = Buy 25%
        # 2 = Buy 50%
        # 3 = Sell 25%
        # 4 = Sell 50%
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )

        # Trading fee (0.1%)
        self.transaction_fee = 0.001

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.balance = self.initial_balance
        self.shares_held = 0

        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance

        observation = self._next_observation()

        return observation, {}

    def _next_observation(self):

        row = self.df.iloc[self.current_step]

        return np.array([
            row["Close"],
            row["RSI"],
            row["MACD"],
            row["SMA20"],
            row["SMA50"],
            row["ATR"],
            row["Returns"],
            row["Volatility"],
            row["BB_Upper"],
            row["BB_Lower"],
            self.balance,
            self.shares_held
        ], dtype=np.float32)

    def step(self, action):

        current_price = self.df.iloc[
            self.current_step
        ]["Close"]

        # HOLD
        if action == 0:
            pass

        # BUY 25%
        elif action == 1:

            cash_to_use = self.balance * 0.25

            shares_to_buy = int(
                cash_to_use // current_price
            )

            if shares_to_buy > 0:

                cost = (
                    shares_to_buy *
                    current_price
                )

                fee = (
                    cost *
                    self.transaction_fee
                )

                if self.balance >= cost + fee:

                    self.balance -= (
                        cost + fee
                    )

                    self.shares_held += (
                        shares_to_buy
                    )

        # BUY 50%
        elif action == 2:

            cash_to_use = self.balance * 0.50

            shares_to_buy = int(
                cash_to_use // current_price
            )

            if shares_to_buy > 0:

                cost = (
                    shares_to_buy *
                    current_price
                )

                fee = (
                    cost *
                    self.transaction_fee
                )

                if self.balance >= cost + fee:

                    self.balance -= (
                        cost + fee
                    )

                    self.shares_held += (
                        shares_to_buy
                    )

        # SELL 25%
        elif action == 3:

            shares_to_sell = int(
                self.shares_held * 0.25
            )

            if shares_to_sell > 0:

                sale_value = (
                    shares_to_sell *
                    current_price
                )

                fee = (
                    sale_value *
                    self.transaction_fee
                )

                self.balance += (
                    sale_value - fee
                )

                self.shares_held -= (
                    shares_to_sell
                )

        # SELL 50%
        elif action == 4:

            shares_to_sell = int(
                self.shares_held * 0.50
            )

            if shares_to_sell > 0:

                sale_value = (
                    shares_to_sell *
                    current_price
                )

                fee = (
                    sale_value *
                    self.transaction_fee
                )

                self.balance += (
                    sale_value - fee
                )

                self.shares_held -= (
                    shares_to_sell
                )

        # Update net worth
        self.net_worth = (
            self.balance +
            self.shares_held *
            current_price
        )

        # Safe log-return reward
        safe_previous = max(
            self.previous_net_worth,
            1e-8
        )

        reward = np.log(
            self.net_worth /
            safe_previous
        )

        # Small hold penalty
        if action == 0:
            reward -= 0.001

        # Drawdown penalty
        drawdown = (
            self.initial_balance -
            self.net_worth
        ) / self.initial_balance

        if drawdown > 0.10:
            reward -= 0.01

        self.previous_net_worth = (
            self.net_worth
        )

        # Episode termination
        terminated = (
            self.current_step >=
            len(self.df) - 1
        )

        # Stop-loss at -10%
        if self.net_worth < (
            self.initial_balance * 0.90
        ):
            reward -= 1.0
            terminated = True

        truncated = False

        if not terminated:

            self.current_step += 1

            observation = (
                self._next_observation()
            )

        else:

            observation = np.zeros(
                self.observation_space.shape,
                dtype=np.float32
            )

        info = {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )