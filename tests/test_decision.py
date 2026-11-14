"""Run with `uv run python -m pytest tests` from the project root."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from lib.decision import (
    best_shift,
    cheapest_with_rest,
    count_rest_breaks,
    flex_check,
    pick_freely,
    pick_near_plan,
    pick_weekly,
    pick_with_rest,
    premium,
    rest_check,
    shift_windows,
    week_mean_premium,
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


def plan_at(starts: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [MONDAY + timedelta(days=d) for d in range(len(starts))],
            "start": starts,
        }
    )


def test_pick_near_plan_moves_only_within_reach():
    # the cheapest shifts start at 14 and 0, the plan says 6 and 6
    frame = days_cheapest_at([14, 0])
    windows = frame.pipe(shift_windows, score(frame))
    plan = plan_at([6, 6])
    both_ways = windows.pipe(pick_near_plan, "price", plan, 2, 2, False)
    earlier_only = windows.pipe(pick_near_plan, "price", plan, 2, 0, False)
    assert both_ways["start"].to_list() == [8, 4]
    assert earlier_only["start"].to_list() == [6, 4]


def test_pick_near_plan_with_no_reach_keeps_the_plan():
    frame = days_cheapest_at([14, 0, 9])
    windows = frame.pipe(shift_windows, score(frame))
    plan = plan_at([6, 3, 16])
    for keep_rest in (False, True):
        picks = windows.pipe(pick_near_plan, "price", plan, 0, 0, keep_rest)
        assert picks["start"].to_list() == [6, 3, 16]


def test_pick_near_plan_keeps_rest_when_the_plan_does():
    # the plan 16, 11 keeps rest; moving day 1 later to 18 is out of range, but day 1
    # at 16 and day 2 wanting 8 would break it, so day 2 stops at 11
    frame = days_cheapest_at([16, 8])
    windows = frame.pipe(shift_windows, score(frame))
    picks = windows.pipe(pick_near_plan, "price", plan_at([16, 11]), 3, 3, True)
    assert picks["start"].to_list() == [16, 11]
    assert count_rest_breaks(picks) == 0
    free = windows.pipe(pick_near_plan, "price", plan_at([16, 11]), 3, 3, False)
    assert free["start"].to_list() == [16, 8]


def test_flex_check_runs_from_the_plan_to_the_final_choice():
    # the plan score is flat, so the plan is the first window, 00:00; the final score
    # is the price itself, so an unlimited reach finds hindsight
    frame = days_cheapest_at([6, 8, 10, 9])
    table = frame.pipe(
        flex_check,
        pl.col("hour") * 0.0,
        pl.col("price_da") + premium(),
        {"none": (0, 0), "unlimited": (16, 16), "later-2": (0, 2)},
    )
    free = table.filter(~pl.col("keep_rest"))
    unlimited = free.filter(pl.col("reach") == "unlimited").row(0, named=True)
    none = free.filter(pl.col("reach") == "none").row(0, named=True)
    later_2 = free.filter(pl.col("reach") == "later-2").row(0, named=True)
    assert unlimited["saving_share"] == 1.0
    assert none["moved_share"] == 0.0
    assert later_2["mean_move_hours"] == 2.0
    assert none["saving_share"] < later_2["saving_share"] < 1.0
    assert table.height == 6


def calendar_days(starts: list[int], first_day: date = MONDAY) -> pl.DataFrame:
    """`days_cheapest_at`, but on real consecutive dates from `first_day` with their own
    weekdays, so weekends carry the premium."""
    frame = days_cheapest_at(starts)
    return frame.with_columns(
        date=pl.col("date") + (first_day - MONDAY),
    ).with_columns(weekday=pl.col("date").dt.weekday())


def test_week_mean_premium_is_the_premium_averaged_over_a_week():
    week = calendar_days([6] * 7)
    by_hour = week.group_by("hour").agg(
        averaged=premium().mean(), week_mean=week_mean_premium().first()
    )
    for row in by_hour.iter_rows(named=True):
        assert row["week_mean"] == pytest.approx(row["averaged"])
    assert week.filter(pl.col("hour") == 12).select(week_mean_premium()).row(0)[
        0
    ] == pytest.approx(10.0 * 2 / 7)


def test_pick_weekly_holds_one_start_a_week_at_the_lowest_weekly_total():
    # four days want 3, three want 10: summed over the week, 3 wins and holds all week
    frame = calendar_days([3, 3, 10, 10, 10, 3, 3], first_day=MONDAY)
    windows = frame.pipe(shift_windows, {"price": pl.col("price_da")})
    picks = windows.pipe(pick_weekly, "price", False)
    assert picks.height == 7
    assert picks["start"].to_list() == [3] * 7
    assert picks.columns == windows.columns


def test_pick_weekly_starts_a_new_week_on_monday():
    # Thursday to Sunday wants 9, the next Monday to Wednesday wants 2
    frame = calendar_days([9, 9, 9, 9, 2, 2, 2], first_day=MONDAY + timedelta(days=3))
    windows = frame.pipe(shift_windows, {"price": pl.col("price_da")})
    picks = windows.pipe(pick_weekly, "price", False)
    assert picks["start"].to_list() == [9, 9, 9, 9, 2, 2, 2]


def test_pick_weekly_keeps_rest_at_the_week_boundary():
    # week one wants 16, week two wants 0; free, Monday 00:00 follows Sunday 16:00
    frame = calendar_days([16] * 7 + [0] * 7)
    windows = frame.pipe(shift_windows, {"price": pl.col("price_da")})
    free = windows.pipe(pick_weekly, "price", False)
    kept = windows.pipe(pick_weekly, "price", True)
    assert count_rest_breaks(free) == 1
    assert count_rest_breaks(kept) == 0
    assert kept["start"].to_list() == [16] * 7 + [11] * 7


def test_rest_check_counts_start_changes_and_picks_weekly_scores_by_week():
    frame = calendar_days([16] * 7 + [0] * 7)
    scores = {
        "daily": pl.col("price_da") + premium(),
        "weekly": pl.col("price_da") + premium(),
    }
    table = frame.pipe(rest_check, scores, weekly=["weekly"])
    weekly = table.filter(pl.col("schedule") == "weekly").row(0, named=True)
    assert weekly["start_changes"] == 1
    assert weekly["start_changes_rest"] == 1
    assert weekly["rest_breaks"] == 1


def test_flex_check_with_a_weekly_plan_and_no_reach_keeps_the_weekly_plan():
    frame = calendar_days([3, 3, 10, 10, 10, 3, 3])
    plan_score = pl.col("price_da")
    table = frame.pipe(
        flex_check,
        plan_score,
        pl.col("price_da") + premium(),
        {"none": (0, 0), "unlimited": (16, 16)},
        weekly_plan=True,
    )
    windows = frame.pipe(shift_windows, {"plan": plan_score})
    # Series.sum() is typed as any number, Decimal included; cost is always a float
    weekly_cost = float(windows.pipe(pick_weekly, "plan", False)["cost"].sum())
    standard = float(windows.filter(pl.col("start") == 6)["cost"].sum())
    hindsight = float(windows.pipe(pick_freely, "cost")["cost"].sum())
    none = table.filter(~pl.col("keep_rest"), pl.col("reach") == "none").row(
        0, named=True
    )
    unlimited = table.filter(~pl.col("keep_rest"), pl.col("reach") == "unlimited")
    assert none["saving_share"] == pytest.approx(
        (standard - weekly_cost) / (standard - hindsight)
    )
    assert unlimited["saving_share"].item() == 1.0


def test_pick_weekly_is_not_fooled_by_a_start_missing_on_the_short_day():
    # the Sunday of the March clock change has no 02:00, so no window starts there.
    # Every day wants 09:00, and a weekly sum over 6 days would make 02:00 look cheaper
    frame = calendar_days([9] * 7).filter(
        ~((pl.col("date") == MONDAY + timedelta(days=6)) & (pl.col("hour") == 2))
    )
    windows = frame.pipe(shift_windows, {"price": pl.col("price_da") + 1000.0})
    picks = windows.pipe(pick_weekly, "price", False)
    assert picks["start"].to_list() == [9] * 7
