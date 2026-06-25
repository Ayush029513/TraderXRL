# TraderXRL

TraderXRL is a Reinforcement Learning based algorithmic trading system built using Python, Gymnasium, and Stable-Baselines3. The project trains a PPO (Proximal Policy Optimization) agent to make trading decisions using technical indicators and market data.

## Features

* Custom Gymnasium trading environment
* PPO reinforcement learning agent
* Technical indicators:

  * RSI
  * MACD
  * SMA20
  * SMA50
  * ATR
  * Bollinger Bands
  * Volatility
* Transaction fee simulation
* Backtesting framework
* Risk analysis
* Performance comparison with Buy & Hold strategy
* Equity curve visualization

## Project Structure

TraderXRL/
├── agents/
├── backtesting/
├── data/
├── env/
├── indicators/
├── models/
├── notebooks/
├── results/

## Performance

![Equity Curve](results/equity_curve.png)

![Performance Comparison](results/performance_comparison.png)


## Installation

```bash
git clone https://github.com/Ayush029513/TraderXRL.git
cd TraderXRL
pip install -r requirements.txt

## Technologies Used

- Python
- Pandas
- NumPy
- Gymnasium
- Stable-Baselines3
- PPO
- Matplotlib
- Technical Analysis (TA)

## Results

- PPO-based Reinforcement Learning Agent
- Custom Trading Environment
- Backtesting Framework
- Risk Analysis (Sharpe Ratio, Drawdown)
- Equity Curve Visualization

## Live Demo
https://traderxrl.streamlit.app