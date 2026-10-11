"""Plumbing for the Energy-Charts API.

Source: https://api.energy-charts.info (Fraunhofer ISE). No API key, CC BY 4.0.
Prices are relabelled from Bundesnetzagentur | SMARD.de; the load and generation
forecasts originate with the four German TSOs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date
from typing import Any

import polars as pl

BASE = "https://api.energy-charts.info"
YEARS = range(2019, date.today().year + 1)
SLEEP_SECONDS = 6.0
RETRIES = 5

_last_call: float = 0.0

PathBuilder = Callable[[str, str], str]
"""Given an inclusive start and exclusive end date, return an API path."""


def get_json(path: str) -> dict[str, Any] | None:
    """One throttled GET. None when the API has no data for the range.

    The API returns 429 within a few unthrottled calls, so requests are spaced out and
    backed off rather than retried immediately.
    """
    global _last_call
    for attempt in range(RETRIES):
        wait = SLEEP_SECONDS - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as response:
                raw = response.read().decode("utf8", "ignore")
            _last_call = time.monotonic()
            return json.loads(raw) if raw.lstrip().startswith("{") else None
        except urllib.error.HTTPError as error:
            _last_call = time.monotonic()
            backoff = SLEEP_SECONDS * (attempt + 2)
            print(f"    HTTP {error.code}, backing off {backoff:.0f}s")
            time.sleep(backoff)
        except Exception as error:  # noqa: BLE001
            _last_call = time.monotonic()
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
