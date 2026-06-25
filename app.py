from dashboard.sidebar import render_sidebar
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="TraderXRL",
    page_icon="📈",
    layout="wide"
)
stock, capital, uploaded_file, run = render_sidebar()

st.title("📈 TraderXRL")
st.subheader("Reinforcement Learning Trading Agent")

# Load equity curve
equity_df = pd.read_csv("results/equity_curve.csv")

st.header("Portfolio Performance")

st.line_chart(equity_df.iloc[:, -1])

st.header("Performance Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Initial Capital", "$10,000")
col2.metric("Agent Value", "$29,919")
col3.metric("Agent Return", "199.2%")

st.header("Equity Curve Image")

st.image("results/equity_curve.png")

st.header("Performance Comparison")

st.image("results/performance_comparison.png")

st.success("TraderXRL PPO Agent Dashboard")