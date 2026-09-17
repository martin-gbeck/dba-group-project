import polars as pl

from spot_price_forecast import paths

CSV_DIR = paths.DATA_DIR / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

pl.read_parquet(paths.PROCESSED_DATA_PATH).write_csv(CSV_DIR / "processed_data.csv")
