from pathlib import Path

DATA_DIR = Path("data")

RAW_DATA_DIR = DATA_DIR / "raw_data"

FORECAST_DATA_PATH = RAW_DATA_DIR / "de_lu_day_ahead_forecasts.parquet"
DAY_AHEAD_PRICE_DATA_PATH = RAW_DATA_DIR / "de_lu_day_ahead_prices.parquet"
NEIGHBOR_PRICES_DATA_PATH = RAW_DATA_DIR / "neighbor_prices.parquet"

PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.parquet"

TRAINING_DATA_PATH = PROCESSED_DATA_DIR / "training_data.parquet"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation_data.parquet"
TESTING_DATA_PATH = PROCESSED_DATA_DIR / "testing_data.parquet"


RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
