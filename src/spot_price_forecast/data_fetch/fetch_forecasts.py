from __future__ import annotations

import polars as pl

from spot_price_forecast.data_fetch.fetching_helpers import fetch_series, join_all, save
from spot_price_forecast.data_schemas import ForecastSchema
from spot_price_forecast.paths import FORECAST_DATA_PATH

PRODUCTION_TYPES: dict[str, str] = {
    "load": "fc_load",
    "solar": "fc_solar",
    "wind_onshore": "fc_wind_onshore",
    "wind_offshore": "fc_wind_offshore",
}


def fetch(production_type: str, column: str) -> pl.DataFrame:
    return fetch_series(
        lambda start, end: (
            f"/public_power_forecast?country=de&production_type={production_type}"
            f"&forecast_type=day-ahead&start={start}&end={end}"
        ),
        # this endpoint returns "forecast_values"; /price returns "price"
        "forecast_values",
        column,
    )


def main() -> None:
    """Pull /public_power_forecast and save it: TSO day-ahead forecasts for Germany, MW.

    Forecasts published ahead of delivery, never actuals: actuals for day D are not
    knowable at the 11:55 decision point, so they are absent by design.

    Timing caveat that belongs in the report: these are computed at 08:00 on D-1 and
    published at 18:00 on D-1, with the auction gate at 12:00 in between. See project.md.

        uv run fetch-forecasts
    """

    df = join_all(
        [
            fetch(
                kind,
                column,
            )
            for kind, column in PRODUCTION_TYPES.items()
        ]
    ).with_columns(pl.col("utc_timestamp").dt.replace_time_zone("UTC"))

    validated_df = ForecastSchema.validate(df)

    save(
        validated_df,
        FORECAST_DATA_PATH,
    )


if __name__ == "__main__":
    main()
