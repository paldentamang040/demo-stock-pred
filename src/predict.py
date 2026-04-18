import numpy as np
from sklearn.metrics import mean_absolute_error


def predict(model, X_test, y_test, target_scaler):
    pred_scaled = model.predict(X_test, verbose=0)
    pred   = target_scaler.inverse_transform(pred_scaled).reshape(-1)
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    return y_true, pred


def compute_metrics(y_true, pred):
    mae  = float(mean_absolute_error(y_true, pred))
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    actual_dir = (y_true > 0).astype(int)
    pred_dir   = (pred   > 0).astype(int)
    da = float((actual_dir == pred_dir).mean())
    return {"mae": mae, "rmse": rmse, "directional_accuracy": da}


def next_day_prediction(model, df_full, target_scaler, window_size=60):
    """Run prediction on the most recent window to get next-day return forecast."""
    from sklearn.preprocessing import MinMaxScaler
    from config import FEATURE_COLS

    df = df_full[FEATURE_COLS].dropna().copy()

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(df.iloc[:int(len(df) * 0.70)][FEATURE_COLS])
    scaled = feature_scaler.transform(df[FEATURE_COLS])

    last_window = scaled[-window_size:].reshape(1, window_size, len(FEATURE_COLS))
    pred_scaled  = model.predict(last_window, verbose=0)
    pred_return  = target_scaler.inverse_transform(pred_scaled).flatten()[0]
    return pred_return
