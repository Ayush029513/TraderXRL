import streamlit as st

def render_sidebar():
    st.sidebar.title("📈 TraderXRL")

    stock = st.sidebar.selectbox(
        "Select Stock",
        ["AAPL", "TSLA", "MSFT", "GOOG"]
    )

    capital = st.sidebar.number_input(
        "Initial Capital",
        value=10000,
        step=1000
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    run = st.sidebar.button("Run Backtest")

    return stock, capital, uploaded_file, run