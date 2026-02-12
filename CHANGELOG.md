# Changelog

All notable changes to this project will be documented in this file.

## [2026-02-11] - Sentiment Engine & UI Integration
* **Merged** branch `shravanFeature` into main, integrating the initial Streamlit `app.py` dashboard framework. (**Contributor: Rushik**)
* **Added** hybrid VADER/FinBERT sentiment logic with global scope functions for successful pickling. (**Contributor: Rushik**)
* **Improved** Enhanced `map_emotion` keywords to include crypto-specific signals like "crash" and "rally". (**Contributor: Rushik**)
* **Fixed** Resolved `EOFError: Ran out of input` by ensuring atomic writes and `KeyError: 'text'` by targeting the `title` column. (**Contributor: Rushik**)

## [2026-02-10] - Analytics & Dashboard Merge
* **Added** `src/sentiment/analyzer.py` to provide a modularized interface for the Streamlit dashboard to calculate real-time market sentiment percentages. (**Contributor: Rushik**)
* **Added** `src/sentiment/models/sentiment_logic.pkl` containing the trained VADER analyzer and emotion mapping logic. (**Contributor: Rushik**)
* **Added** `src/sentiment/Train_sentiment_.py` to implement hybrid VADER/FinBERT sentiment logic with global scope functions for successful pickling. (**Contributor: Rushik**)
* **Added** `src/sentiment/test_sentiment.py` to validate model accuracy against GitHub news data and generate market psychology distributions. (**Contributor: Rushik**)
* **Refactored** original NLP logic from `NLP.py` into a modularized system: `Train_sentiment_.py`, `test_sentiment.py`, and `analyzer.py`. (**Contributor: Rushik**)
* **Added** ARIMA forecasting results for multiple coins including BTC, ETH, SOL, USDC, and USDT. (**Contributor: Shravan Shidruk**)
* **Added** Walk-forward validation plots and overview visualizations in `Output/Arima/`. (**Contributor: Shravan Shidruk**)
* **Added** `notebooks/USDT_INR/first_attempt.ipynb` for initial Tether price analysis. (**Contributor: Shravan Shidruk**)
* **Changed** `requirements.txt` updated to track new dependencies for data science modules. (**Contributor: Shravan Shidruk**)

## [2026-02-04] - Initial Sentiment Logic (nlp branch)
* **Added** `NLP.py` containing the initial prototype for sentiment analysis and text processing. (**Contributor: stefiegeorge**)
* **Created** Logic foundations for categorizing financial headlines which served as the basis for the current sentiment pipeline. (**Contributor: stefiegeorge**)

## [2026-02-02] - Data Pipeline Refactoring
- **Added** `src/api/fetch_news.py` for collecting cleaned bitcoin news. (**Contributor: Rushik**)

## [2026-01-21] - Path & Performance Optimization
- **Added** `src/config.py` for centralized path management. (**Contributor: Rushik**)
- **Fixed** Resolved `YFRateLimitError` via batch downloads in `fetch_data.py`. (**Contributor: Rushik**)
- **Changed** `clean_data.py` now converts prices to INR using live exchange rates. (**Contributor: Rushik**)

## [2026-01-20] - Initial Data Setup
- **Added** Basic `fetch_data.py` script. (**Contributor: Rushik**)
- **Created** Folder structure for `data/raw` and `data/processed`. (**Contributor: Rushik**)