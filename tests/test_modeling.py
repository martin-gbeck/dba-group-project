"""Run with `uv run --with pytest python -m pytest tests` from the project root."""

from datetime import date, timedelta

import polars as pl

from lib.modeling import trailing_hourly_mean

FIRST = date(2025, 1, 1)


def prices_by_day(values: list[float], hours: int = 2) -> pl.DataFrame:
    """One price per day and hour: day d costs values[d] in every hour."""
    return pl.DataFrame(
        [
            {"date": FIRST + timedelta(days=d), "hour": hour, "price": value}
            for d, value in enumerate(values)
            for hour in range(hours)
        ]
    )


def mean_on(table: pl.DataFrame, day: int, hour: int = 0) -> float:
    # item() is typed as Any; mean_price is always a float
    return float(
        table.filter(
            pl.col("date") == FIRST + timedelta(days=day), pl.col("hour") == hour
        )["mean_price"].item()
    )


def test_trailing_mean_covers_the_window_ending_on_the_date():
    table = prices_by_day([1, 2, 3, 4, 5]).pipe(trailing_hourly_mean, 3)
    assert mean_on(table, 4) == 4.0  # days 2, 3, 4
    assert mean_on(table, 2, hour=1) == 2.0  # days 0, 1, 2


def test_trailing_mean_steps_back_whole_weeks():
    values = [float(d) for d in range(15)]
    table = prices_by_day(values).pipe(trailing_hourly_mean, 2, every=7)
    assert mean_on(table, 14) == 10.5  # days 14 and 7


def test_trailing_mean_skips_a_missing_hour():
    # the March clock change has no hour 2, so that day drops out at hour 2
    prices = prices_by_day([1, 2, 3], hours=3).filter(
        ~((pl.col("date") == FIRST + timedelta(days=1)) & (pl.col("hour") == 2))
    )
    table = prices.pipe(trailing_hourly_mean, 3)
    assert mean_on(table, 2, hour=2) == 2.0  # days 0 and 2
    assert mean_on(table, 2, hour=1) == 2.0  # days 0, 1 and 2
