"""The three raw datasets, pulled from the Energy-Charts API and saved to data/ as parquet.

Source: https://api.energy-charts.info (Fraunhofer ISE). No API key, CC BY 4.0.
Prices are relabelled from Bundesnetzagentur | SMARD.de; the load and generation
forecasts originate with the four German TSOs.

Nothing here processes the data: no resampling, no joining across datasets, no dropped
rows, no validation. Each file keeps the resolution the API serves, hourly until the
European market-time-unit change on 2025-10-01 and 15-minute after it (the forecasts
are 15-minute throughout). All processing lives in the notebook.

Each fetch function reads its saved file when one exists. `refresh=True` pulls from the
API again and overwrites it. Calls are spaced 6 seconds apart and the API still
rate-limits, so most of a pull is waiting: on 2026-10-11 the three took 2, 8 and 25
minutes.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

BASE = "https://api.energy-charts.info"
YEARS = range(2019, date.today().year + 1)
SLEEP_SECONDS = 6.0
RETRIES = 5

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

last_call: float = 0.0

PathBuilder = Callable[[str, str], str]
"""Given an inclusive start and exclusive end date, return an API path."""


def get_json(path: str) -> dict[str, Any] | None:
    """One throttled GET. None when the API has no data for the range.

    The API returns 429 within a few unthrottled calls, so requests are spaced out and
    backed off rather than retried immediately.
    """
    global last_call
    for attempt in range(RETRIES):
        wait = SLEEP_SECONDS - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as response:
                raw = response.read().decode("utf8", "ignore")
            last_call = time.monotonic()
            return json.loads(raw) if raw.lstrip().startswith("{") else None
        except urllib.error.HTTPError as error:
            last_call = time.monotonic()
            backoff = SLEEP_SECONDS * (attempt + 2)
            print(f"    HTTP {error.code}, backing off {backoff:.0f}s")
            time.sleep(backoff)
        except Exception as error:  # noqa: BLE001
            last_call = time.monotonic()
            print(f"    {error}")
            time.sleep(SLEEP_SECONDS)
    return None


def fetch_series(make_path: PathBuilder, value_key: str, column: str) -> pl.DataFrame:
    """Pull one series year by year and stack it into utc_timestamp + column.

    The API will not serve eight years in one call. Neighbouring years can both return
    the boundary timestamp, so the stack keeps the first copy of each.
    """
    frames: list[pl.DataFrame] = []
    for year in YEARS:
        print(f"  {column} {year}")
        payload = get_json(make_path(f"{year}-01-01", f"{year + 1}-01-01"))
        if payload is None:
            continue
        seconds, values = payload.get("unix_seconds"), payload.get(value_key)
        if not seconds or values is None:
            continue
        frames.append(
            pl.DataFrame(
                {
                    "utc_timestamp": pl.from_epoch(
                        pl.Series(seconds), time_unit="s"
                    ).dt.replace_time_zone("UTC"),
                    column: pl.Series(values, dtype=pl.Float64),
                }
            )
        )
    if not frames:
        return pl.DataFrame({"utc_timestamp": []})
    return (
        pl.concat(frames, how="diagonal")
        .sort("utc_timestamp")
        .unique(subset="utc_timestamp", keep="first", maintain_order=True)
    )


def fetch_price(zone: str, column: str) -> pl.DataFrame:
    """Day-ahead clearing price for one bidding zone, EUR/MWh."""
    return fetch_series(
        lambda start, end: f"/price?bzn={zone}&start={start}&end={end}", "price", column
    )


def fetch_forecast(production_type: str, column: str) -> pl.DataFrame:
    """One TSO day-ahead forecast for Germany, MW."""
    return fetch_series(
        lambda start, end: (
            f"/public_power_forecast?country=de&production_type={production_type}"
            f"&forecast_type=day-ahead&start={start}&end={end}"
        ),
        # this endpoint returns "forecast_values"; /price returns "price"
        "forecast_values",
        column,
    )


def join_all(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Outer-join several series on utc_timestamp, so no row from any series is lost."""
    usable = [frame for frame in frames if frame.height > 0]
    if not usable:
        return pl.DataFrame({"utc_timestamp": []})
    out = usable[0]
    for frame in usable[1:]:
        out = out.join(frame, on="utc_timestamp", how="full", coalesce=True)
    return out.sort("utc_timestamp")


def load_or_pull(
    path: Path, pull: Callable[[], pl.DataFrame], refresh: bool
) -> pl.DataFrame:
    """Read the parquet file at `path`. When it is missing or `refresh` is set, call
    `pull` and write its result there first."""
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    df = pull()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"wrote {path.name}  rows {df.height:,}  columns {df.width}")
    return df


def fetch_de_prices(refresh: bool = False) -> pl.DataFrame:
    """DE-LU day-ahead clearing price, EUR/MWh: the target. About 2 minutes to pull."""
    return load_or_pull(
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
    return load_or_pull(
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
    return load_or_pull(
        DATA_DIR / "neighbor_prices.parquet",
        lambda: join_all([fetch_price(zone, f"price_{zone}") for zone in NEIGHBORS]),
        refresh,
    )
