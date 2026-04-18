import os

# Try .env file first (local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Resolve API key from env or Streamlit secrets
def _get_api_key():
    # 1. Environment variable (from .env locally)
    key = os.getenv("MARKETAUX_API_KEY")
    if key:
        return key
    # 2. Streamlit secrets (on Streamlit Cloud)
    try:
        import streamlit as st
        return st.secrets["MARKETAUX_API_KEY"]
    except Exception:
        pass
    # 3. Nothing found — raise a clear error
    raise ValueError(
        "MARKETAUX_API_KEY not found. "
        "Add it to .env locally or Streamlit Cloud secrets."
    )

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
WINDOW_SIZE = 60
MARKETAUX_API_KEY = _get_api_key()
FEATURE_COLS = ["Close", "returns", "sentiment_ma_5"]
TARGET_COL = "returns"
