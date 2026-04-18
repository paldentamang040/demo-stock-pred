# Stock Return Forecaster — Streamlit UI

## Setup

1. Copy your saved models into the `saved_models/` folder:
   ```
   saved_models/
   ├── AAPL_best_model.keras
   ├── MSFT_best_model.keras
   ├── GOOGL_best_model.keras
   ├── AMZN_best_model.keras
   └── NVDA_best_model.keras
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

## Project Structure

```
stock_app/
├── app.py              # Streamlit UI
├── config.py           # Symbols, window size, API key
├── requirements.txt
├── saved_models/       # Place your .keras files here
└── src/
    ├── data.py         # Price + sentiment fetching, sequence preparation
    └── predict.py      # Model inference + metrics
```
