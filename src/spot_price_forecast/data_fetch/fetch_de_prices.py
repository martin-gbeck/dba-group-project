from __future__ import annotations

from spot_price_forecast.data_fetch.fetching_helpers import fetch_price, save
from spot_price_forecast.data_schemas import ForecastSchema
from spot_price_forecast.paths import DAY_AHEAD_PRICE_DATA_PATH

GERMANY = "DE-LU"


def main() -> None:
    """Pull /price for DE-LU and save it: the target.

    The German-Luxembourg day-ahead clearing price, EUR/MWh, one value per settlement
    period. This is the series the project predicts.

        uv run fetch-de-prices
    """

    df = fetch_price(GERMANY, "price_da")
    validated_df = ForecastSchema.validate(df)

    save(
        validated_df,
        DAY_AHEAD_PRICE_DATA_PATH,
    )


if __name__ == "__main__":
    main()
