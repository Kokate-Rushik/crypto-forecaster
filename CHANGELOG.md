# Project Update Log

## [2026-01-21] - Data Pipeline Refactoring
- **Added** `src/config.py` for centralized path management.
- **Fixed** Resolved `YFRateLimitError` by implementing batch downloads in `fetch)data.py`.
- **Changed** `clean_data.py` now converts prices to INR using live exchanges rates.

## [2026-01-20] - Initial Data Setup
- **Added** Basic `fetch_data.py` script.
- **Created** Folder structure for `data/raw` and `data/processed`

