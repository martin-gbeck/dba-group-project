"""The business decision: when tomorrow's 8-hour shift starts, and what each hour costs.

Every function works on the hourly table built in notebook section 4.2, which carries
`date`, `hour` and `weekday`.
"""

import polars as pl

SHIFT_HOURS = 8
SHIFT_MW = 62.5  # 500 MWh a day over 8 hours
# extra staff cost per hour worked from 22:00 to 06:00 or at the weekend:
# about 25% on a crew costing 2,500 EUR an hour
PREMIUM_EUR_PER_HOUR = 625.0
STANDARD_SHIFT = (6, 13)  # first and last hour of the 06:00 to 14:00 reference shift


def premium() -> pl.Expr:
    """The staff premium of each hour in EUR/MWh: 625 EUR over 62.5 MWh at night and weekends."""
    night_or_weekend = (
        (pl.col("hour") >= 22) | (pl.col("hour") < 6) | (pl.col("weekday") >= 6)
    )
    return (
        pl.when(night_or_weekend).then(PREMIUM_EUR_PER_HOUR / SHIFT_MW).otherwise(0.0)
    )


def best_shift(score: pl.Expr) -> pl.Expr:
    """True for the 8 consecutive hours of each day whose scores add up to the least."""
    position = pl.int_range(pl.len()).over("date")  # 0 for the first hour of the day

    # total over the 8 hours ending at each position. The first 7 hours have no full
    # window; they get infinity rather than null so arg_min counts positions correctly
    window_total = score.rolling_sum(SHIFT_HOURS).over("date").fill_null(float("inf"))

    # arg_min returns an unsigned integer, which would wrap below zero in shift_end - 8
    shift_end = window_total.arg_min().over("date").cast(pl.Int64)

    # a day with any missing score cannot be scheduled by that rule
    complete_day = score.is_not_null().all().over("date")

    return (position <= shift_end) & (position > shift_end - SHIFT_HOURS) & complete_day
