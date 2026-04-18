import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
from config import MARKETAUX_API_KEY, WINDOW_SIZE, FEATURE_COLS, TARGET_COL


def fetch_price_data(symbol, period="2y"):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).astype("datetime64[ns]")
    df.index = df.index.normalize()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip("_") for col in df.columns.to_flat_index()]
    df["Close"] = df[[c for c in df.columns if "Close" in c][0]]
    df["returns"] = df["Close"].pct_change()
    return df.dropna().copy()


def fetch_sentiment(symbol, start_date, end_date, limit=100):
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "api_token": MARKETAUX_API_KEY,
        "symbols": symbol,
        "language": "en",
        "must_have_entities": "true",
        "published_after": pd.Timestamp(start_date).strftime("%Y-%m-%dT00:00:00"),
        "published_before": pd.Timestamp(end_date).strftime("%Y-%m-%dT23:59:59"),
        "limit": limit,
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return pd.DataFrame(columns=["news_date", "sentiment_raw"])

    rows = []
    for article in payload.get("data", []):
        pub_date = article.get("published_at")
        entities = article.get("entities", [])
        if pub_date is None:
            continue
        dt = pd.to_datetime(pub_date, utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        dt = dt.tz_localize(None).astype("datetime64[ns]")
        dt = dt.normalize()
        scores = [float(e["sentiment_score"]) for e in entities
                  if e.get("symbol") == symbol and e.get("sentiment_score") is not None]
        if scores:
            rows.append({"news_date": dt, "sentiment_raw": float(np.mean(scores))})

    if not rows:
        return pd.DataFrame(columns=["news_date", "sentiment_raw"])
    return pd.DataFrame(rows).sort_values("news_date")


def build_features(symbol, period="2y"):
    df = fetch_price_data(symbol, period)
    news_df = fetch_sentiment(symbol, df.index.min(), df.index.max())

    trade_df = df.reset_index().rename(columns={"index": "trade_date", "Date": "trade_date"})
    trade_df["trade_date"] = pd.to_datetime(trade_df["trade_date"]).dt.normalize()
    trade_df = trade_df.sort_values("trade_date")

    if news_df.empty:
        trade_df["sentiment_raw"] = 0.0
    else:
        daily_news = (news_df.groupby("news_date")
                      .agg(sentiment_raw=("sentiment_raw", "mean"))
                      .reset_index().sort_values("news_date"))
        trade_df["trade_date"] = pd.to_datetime(trade_df["trade_date"]).astype("datetime64[ns]")
        daily_news["news_date"] = pd.to_datetime(daily_news["news_date"]).astype("datetime64[ns]"
        aligned = pd.merge_asof(trade_df.sort_values("trade_date"), daily_news,
                                left_on="trade_date", right_on="news_date", direction="backward")
        trade_df["sentiment_raw"] = aligned["sentiment_raw"].fillna(0.0)

    trade_df["sentiment_lag1"] = trade_df["sentiment_raw"].shift(1).fillna(0.0)
    trade_df["sentiment_ma_5"] = trade_df["sentiment_lag1"].rolling(5, min_periods=1).mean()
    trade_df = trade_df.set_index("trade_date").sort_index()

    return trade_df


def prepare_sequences(df, window_size=WINDOW_SIZE):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    needed = list(dict.fromkeys(FEATURE_COLS + [TARGET_COL]))
    df = df[needed].dropna()

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()

    train_feat = feature_scaler.fit_transform(train_df[FEATURE_COLS])
    val_feat   = feature_scaler.transform(val_df[FEATURE_COLS])
    test_feat  = feature_scaler.transform(test_df[FEATURE_COLS])

    train_tgt = target_scaler.fit_transform(train_df[TARGET_COL].to_numpy().reshape(-1, 1))
    val_tgt   = target_scaler.transform(val_df[TARGET_COL].to_numpy().reshape(-1, 1))
    test_tgt  = target_scaler.transform(test_df[TARGET_COL].to_numpy().reshape(-1, 1))

    def make_seq(feat, tgt):
        X, y = [], []
        for i in range(window_size, len(feat)):
            X.append(feat[i - window_size:i])
            y.append(tgt[i])
        return np.array(X), np.array(y)

    test_feat_ctx = np.vstack([val_feat[-window_size:], test_feat])
    test_tgt_ctx  = np.vstack([val_tgt[-window_size:],  test_tgt])
    X_test, y_test = make_seq(test_feat_ctx, test_tgt_ctx)

    return X_test, y_test, target_scaler, test_df, df
