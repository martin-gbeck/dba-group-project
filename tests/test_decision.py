"""Run with `uv run python -m pytest tests` from the project root."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from lib.decision import (
    best_shift,
    cheapest_with_rest,
    count_rest_breaks,
    pick_freely,
    pick_with_rest,
    premium,
    rest_check,
    shift_windows,
)

MONDAY = date(2025, 1, 6)


def days_cheapest_at(
    starts: list[int], hours: int = 24, other_prices: list[float] | None = None
) -> pl.DataFrame:
    """Weekday hours, one day per start, priced 0 for the eight hours from that day's
    start and `other_prices[d]` (100 by default) in every other hour, so the cheapest
    shift starts there."""
    other_prices = other_prices or [100.0] * len(starts)
    rows = []
    for d, start in enumerate(starts):
        day = MONDAY + timedelta(days=d % 5)  # weekdays only, so no weekend premium
        for hour in range(hours):
            rows.append(
                {
                    "utc_timestamp": datetime(2025, 1, 1)
                    + timedelta(hours=24 * d + hour),
                    "date": MONDAY + timedelta(days=d),
                    "weekday": day.isoweekday(),
                    "hour": hour,
                    "price_da": 0.0 if start <= hour < start + 8 else other_prices[d],
                }
            )
    return pl.DataFrame(rows)


def score(frame: pl.DataFrame) -> dict[str, pl.Expr]:
    return {"price": pl.col("price_da") + premium()}


def test_free_pick_matches_best_shift():
    frame = days_cheapest_at([14, 3, 9, 0, 16])
    picks = frame.pipe(shift_windows, score(frame)).pipe(pick_freely, "price")
    expected = (
        frame.filter(best_shift(pl.col("price_da") + premium()))
        .group_by("date")
        .agg(start=pl.col("hour").min())
        .sort("date")
    )
    assert picks["start"].to_list() == expected["start"].to_list()


def test_count_rest_breaks_flags_starts_more_than_five_hours_earlier():
    picks = pl.DataFrame(
        {
            "date": [MONDAY + timedelta(days=d) for d in range(4)],
            "start": [14, 9, 3, 16],
        }
    )
    # 14 -> 9 leaves exactly 11 hours, 9 -> 3 leaves 10, 3 -> 16 leaves 29
    assert count_rest_breaks(picks) == 1


def test_count_rest_breaks_skips_a_gap_in_the_dates():
    picks = pl.DataFrame(
        {"date": [MONDAY, MONDAY + timedelta(days=2)], "start": [16, 0]}
    )
    assert count_rest_breaks(picks) == 0


def test_greedy_keeps_the_first_day_free_and_respects_rest_after():
    frame = days_cheapest_at([16, 0, 0])
    picks = frame.pipe(shift_windows, score(frame)).pipe(pick_with_rest, "price")
    starts = picks["start"].to_list()
    assert starts[0] == 16
    assert starts[1] == 11  # the earliest start 11 hours after a 16:00 to 24:00 shift
    assert count_rest_breaks(picks) == 0


def test_dynamic_programme_beats_greedy_when_today_should_give_way():
    # day 1 at 16:00 forces day 2 to start at 11:00 or later. Day 1's other hours are
    # nearly as cheap and day 2's are dear, so starting day 1 earlier costs less over
    # both days, which only a plan over the whole period sees
    frame = days_cheapest_at([16, 0], other_prices=[1.0, 100.0])
    windows = frame.pipe(shift_windows, score(frame))
    greedy_cost = windows.pipe(pick_with_rest, "cost")["cost"].sum()
    planned = windows.pipe(cheapest_with_rest)
    assert planned["cost"].sum() < greedy_cost
    assert count_rest_breaks(planned) == 0


def test_dynamic_programme_equals_free_hindsight_when_no_rule_binds():
    frame = days_cheapest_at([6, 8, 10, 9])
    windows = frame.pipe(shift_windows, score(frame))
    assert windows.pipe(cheapest_with_rest)["start"].to_list() == [6, 8, 10, 9]


def test_clock_change_day_gives_one_window_per_position():
    # a 25-hour day has 18 eight-hour windows, a 23-hour day 16
    long_day = days_cheapest_at([6], hours=25)
    short_day = days_cheapest_at([6], hours=23)
    assert long_day.pipe(shift_windows, score(long_day)).height == 18
    assert short_day.pipe(shift_windows, score(short_day)).height == 16


def test_rest_check_hindsight_scores_one_and_rule_costs_nothing_when_unbinding():
    frame = days_cheapest_at([6, 8, 10, 9])
    table = frame.pipe(rest_check, score(frame))
    hindsight = table.filter(pl.col("schedule") == "hindsight").row(0, named=True)
    assert hindsight["saving_share"] == 1.0
    assert hindsight["saving_share_rest"] == 1.0
    assert hindsight["rest_eur_per_year"] == 0.0
    assert hindsight["rest_breaks"] == 0


def test_clock_change_days_use_their_real_hour_labels():
    # 30 Mar 2025 skips 02:00, 26 Oct 2025 has 02:00 twice
    frame = days_cheapest_at([6, 6], hours=24)
    short_hours = [0, 1, *range(3, 24)]
    long_hours = [0, 1, 2, 2, *range(3, 24)]
    frame = pl.concat(
        [
            frame.filter(pl.col("date") == MONDAY)
            .head(23)
            .with_columns(hour=pl.Series(short_hours)),
            days_cheapest_at([6], hours=25).with_columns(
                date=pl.lit(MONDAY + timedelta(days=1)),
                utc_timestamp=pl.col("utc_timestamp") + timedelta(days=1),
                hour=pl.Series(long_hours),
            ),
        ]
    )
    windows = frame.pipe(shift_windows, score(frame))
    starts = windows.group_by("date", maintain_order=True).agg(
        first=pl.col("start").first(), last=pl.col("start").last()
    )
    assert starts["first"].to_list() == [0, 0]
    assert starts["last"].to_list() == [16, 16]
    # the 06:00 window exists once on each day
    assert windows.filter(pl.col("start") == 6).height == 2


def test_rest_check_refuses_a_day_with_missing_scores():
    # prices are complete, the forecast is missing for the first day
    frame = days_cheapest_at([6, 8]).with_columns(
        pred=pl.when(pl.col("date") == MONDAY).then(None).otherwise(pl.col("price_da"))
    )
    with pytest.raises(ValueError, match="missing"):
        frame.pipe(rest_check, {"model": pl.col("pred") + premium()})


def test_rest_check_refuses_a_gap_in_the_dates():
    frame = days_cheapest_at([6, 8, 10]).filter(
        pl.col("date") != MONDAY + timedelta(days=1)
    )
    with pytest.raises(ValueError, match="consecutive"):
        frame.pipe(rest_check, score(frame))
