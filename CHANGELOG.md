# Project Update Log

## [2026-02-11] - Sentiment Engine & UI Integration

* **Merged** branch `shravanFeature` into main, integrating the initial Streamlit `app.py` dashboard framework.
* **Added** `src/sentiment/Train_sentiment_.py` to implement hybrid VADER/FinBERT sentiment logic with global scope functions for successful pickling.
* **Added** `src/sentiment/test_sentiment.py` to validate model accuracy against GitHub news data and generate market psychology distributions.
* **Added** `src/sentiment/models/sentiment_logic.pkl` containing the trained VADER analyzer and emotion mapping logic.
* **Fixed** Resolved `EOFError: Ran out of input` by ensuring atomic writes to the pickle file and verifying non-zero file sizes before loading.
* **Fixed** Resolved `KeyError: 'text'` by updating the news processing logic to target the correct `title` column in the data source.
* **Improved** Enhanced `map_emotion` keywords to include crypto-specific signals like "crash", "rout", "pullback", and "rally".

## [2026-02-10] - Analytics & Dashboard ('shravanFeature' branch)

* **Added** ARIMA forecasting results for multiple coins including BTC, ETH, SOL, USDC, and USDT.
* **Added** Walk-forward validation plots and overview visualizations in `Output/Arima/` for real-time performance tracking.
* **Added** `notebooks/USDT_INR/first_attempt.ipynb` for initial Tether price analysis.
* **Changed** `requirements.txt` updated to track new dependencies for the data science and web interface modules.

## [2026-02-02] - Data Pipeline Refactoring
- **Added** `src/api/fetch_news.py` for collecting cleaned bitcoin news.

## [2026-01-21] - Data Pipeline Refactoring
- **Added** `src/config.py` for centralized path management.
- **Fixed** Resolved `YFRateLimitError` by implementing batch downloads in `fetch)data.py`.
- **Changed** `clean_data.py` now converts prices to INR using live exchanges rates.

## [2026-01-20] - Initial Data Setup
- **Added** Basic `fetch_data.py` script.
- **Created** Folder structure for `data/raw` and `data/processed`

