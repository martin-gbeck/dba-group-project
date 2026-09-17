from __future__ import annotations

import polars as pl

from spot_price_forecast import paths
from spot_price_forecast.data_prep.helpers import to_hourly


def main() -> None:
    """Build the modelling table from the three raw datasets.

    uv run build-processed
    """

    lf_price = pl.scan_parquet(paths.DAY_AHEAD_PRICE_DATA_PATH).pipe(to_hourly)
    lf_forecast = pl.scan_parquet(paths.FORECAST_DATA_PATH).pipe(to_hourly)
    lf_neighbor_prices = pl.scan_parquet(paths.NEIGHBOR_PRICES_DATA_PATH).pipe(
        to_hourly
    )

    table = (
        lf_price.join(
            lf_forecast,
            on="utc_timestamp",
            how="left",
        )
        .join(
            lf_neighbor_prices,
            on="utc_timestamp",
            how="left",
        )
        # the forecasts run ahead of the auction, so the tail has no target to learn from
        .drop_nulls("price_da")
    )

    table.sink_parquet(paths.PROCESSED_DATA_PATH)
    # print(
    #     f"wrote {paths.PROCESSED_DATA_PATH}  rows {table.height:,}  columns {table.width}"
    # )


if __name__ == "__main__":
    main()
