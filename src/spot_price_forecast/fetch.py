"""The three raw datasets, pulled from Energy-Charts and saved to data/ as parquet.

Nothing here processes the data: no resampling, no joining across datasets, no dropped
rows, no validation. Each file keeps the resolution the API serves, hourly until the
European market-time-unit change on 2025-10-01 and 15-minute after it (the forecasts
are 15-minute throughout). All processing lives in the notebook.

Each function reads its saved file when one exists. `refresh=True` pulls from the API
again and overwrites it. Calls are spaced 6 seconds apart and the API still rate-limits,
so most of a pull is waiting: on 2026-10-11 the three took 2, 8 and 25 minutes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl

from spot_price_forecast.energy_charts import fetch_forecast, fetch_price, join_all

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

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

FORECASTS: dict[str, str] = {
    "load": "fc_load",
    "solar": "fc_solar",
    "wind_onshore": "fc_wind_onshore",
    "wind_offshore": "fc_wind_offshore",
}


def _saved(path: Path, pull: Callable[[], pl.DataFrame], refresh: bool) -> pl.DataFrame:
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    df = pull()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"wrote {path.name}  rows {df.height:,}  columns {df.width}")
    return df


def fetch_de_prices(refresh: bool = False) -> pl.DataFrame:
    """DE-LU day-ahead clearing price, EUR/MWh: the target. About 2 minutes to pull."""
    return _saved(
        DATA_DIR / "de_lu_day_ahead_prices.parquet",
        lambda: fetch_price("DE-LU", "price_da"),
        refresh,
    )


def fetch_forecasts(refresh: bool = False) -> pl.DataFrame:
    """TSO day-ahead forecasts for Germany, MW. About 8 minutes to pull.

    Forecasts, never actuals: actuals for day D are not knowable at the 11:55 decision
    point. Timing caveat for the report: computed at 08:00 on D-1, published at 18:00
    on D-1, with the auction gate at 12:00 in between. See project.md.
    """
    return _saved(
        DATA_DIR / "de_lu_day_ahead_forecasts.parquet",
        lambda: join_all(
            [fetch_forecast(kind, col) for kind, col in FORECASTS.items()]
        ),
        refresh,
    )


def fetch_neighbor_prices(refresh: bool = False) -> pl.DataFrame:
    """Day-ahead prices of the 11 zones coupled to DE-LU, EUR/MWh. About 25 minutes.

    Only usable lagged: day D's neighbour prices clear in the same auction as ours.
    """
    return _saved(
        DATA_DIR / "neighbor_prices.parquet",
        lambda: join_all([fetch_price(zone, f"price_{zone}") for zone in NEIGHBORS]),
        refresh,
    )
