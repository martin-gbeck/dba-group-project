"""Run with `uv run --with pytest python -m pytest tests` from the project root."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from lib.evaluation import fold_sd_lead, permutation_importance
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


def test_fold_sd_lead_is_the_mean_gap_against_the_larger_fold_sd():
    gap, bar = fold_sd_lead([0.90, 0.94], [0.91, 0.92])
    assert gap == pytest.approx(0.005)
    # the first option's SD, sqrt(2) * 0.02, is the larger of the two
    assert bar == pytest.approx(0.04 / 2**0.5)


class FirstColumn:
    """A stand-in model that predicts the first input column as the price."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0]


def two_years_of_hours() -> pl.DataFrame:
    """Weekday hours over 2023 and 2024, cheapest at a different hour each day, with a
    `signal` column equal to the price and a `noise` column the model never reads."""
    rows = []
    for year in (2023, 2024):
        for d in range(20):
            day = date(year, 1, 2) + timedelta(days=d)
            for hour in range(24):
                price = 0.0 if (d % 17) <= hour < (d % 17) + 8 else 100.0 + hour
                rows.append(
                    {
                        "date": day,
                        "year": year,
                        "hour": hour,
                        "weekday": 1,
                        "price_da": price,
                        "price_lag24": 50.0,
                        "signal": price,
                        "noise": float(hour * d),
                    }
                )
    return pl.DataFrame(rows)


def test_permutation_importance_charges_only_the_column_the_model_reads():
    table = permutation_importance(
        FirstColumn(), ["signal", "noise"], two_years_of_hours(), {"fold-1": 2024}
    )
    rows = {row["column"]: row for row in table.iter_rows(named=True)}
    assert table["column"].to_list() == ["signal", "noise"]
    assert rows["signal"]["saving_share_lost"] > 0.5
    assert rows["signal"]["mae_added"] > 0.0
    assert rows["noise"]["saving_share_lost"] == 0.0
    assert rows["noise"]["mae_added"] == 0.0
