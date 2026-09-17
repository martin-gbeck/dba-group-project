from __future__ import annotations

from spot_price_forecast.data_fetch.fetching_helpers import fetch_price, join_all, save
from spot_price_forecast.data_schemas import NeighborPriceSchema
from spot_price_forecast.paths import NEIGHBOR_PRICES_DATA_PATH

# Coupled with DE-LU. The ring reaches past the immediate borders because Nordic hydro
# (NO2, SE4) and Czech thermal both move German prices through the interconnectors.
NEIGHBORS: tuple[str, ...] = (
    "FR",
    "NL",
    "DK1",
    "DK2",
    "AT",
    "PL",
    "CH",
    "BE",
    "CZ",
    "SE4",
    "NO2",
)


def main() -> None:
    """Pull /price for the zones coupled to DE-LU and save it: predictors.

    One algorithm clears every coupled zone simultaneously, subject to cross-border
    transmission capacity, so a neighbour's settled price carries information about
    DE-LU. Only usable lagged, because day D's neighbour prices are decided in the same
    auction as ours and are not knowable at the gate.

        uv run fetch-neighbor-prices
    """

    df = join_all([fetch_price(zone, f"price_{zone}") for zone in NEIGHBORS])
    validated_df = NeighborPriceSchema.validate(df)

    save(
        validated_df,
        NEIGHBOR_PRICES_DATA_PATH,
    )


if __name__ == "__main__":
    main()
