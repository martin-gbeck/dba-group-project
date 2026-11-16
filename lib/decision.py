"""The business decision: when tomorrow's 8-hour shift starts, and what each hour costs.

Every function works on the hourly table built in notebook section 4.2, which carries
`date`, `hour` and `weekday`.
"""

from collections.abc import Collection

import numpy as np
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


def week_mean_premium() -> pl.Expr:
    """The staff premium of each clock hour averaged over the seven days of a week, in
    EUR/MWh: the full premium from 22:00 to 06:00, and two sevenths of it by day, which
    is the weekend's share. Added to a price profile that stands for a whole week, it
    picks the start that is cheapest over that week."""
    night = (pl.col("hour") >= 22) | (pl.col("hour") < 6)
    full = PREMIUM_EUR_PER_HOUR / SHIFT_MW
    return pl.when(night).then(full).otherwise(full * 2 / 7)


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


def daily_costs(frame: pl.DataFrame) -> pl.DataFrame:
    """Each day's cost in EUR, electricity plus premium for 500 MWh, of four shifts:
    `standard` (06:00 to 14:00), `model` (lowest forecast `pred` plus premium), `rule`
    (lowest forecast residual load) and `hindsight` (lowest cleared price plus premium).
    One row per date, in date order."""
    shifts = ["standard", "model", "rule", "hindsight"]
    return (
        frame.with_columns(
            cost=(pl.col("price_da") + premium())
            * SHIFT_MW,  # EUR for the hour's 62.5 MWh
            standard=pl.col("hour").is_between(*STANDARD_SHIFT),
            model=best_shift(pl.col("pred") + premium()),
            rule=best_shift(pl.col("fc_residual")),
            hindsight=best_shift(pl.col("price_da") + premium()),
        )
        .group_by("date")
        .agg(
            [
                pl.col("cost").filter(pl.col(shift)).sum().alias(shift)
                for shift in shifts
            ]
        )
        .sort("date")
    )


# Arbeitszeitgesetz §5: at least 11 hours off between the end of one shift and the
# start of the next. With 8-hour shifts a day apart, a start may be at most 5 hours
# earlier than the day before's
MIN_REST_HOURS = 11
EARLIEST_START_AFTER = SHIFT_HOURS + MIN_REST_HOURS - 24  # -5: start_next - start_prev


def shift_windows(frame: pl.DataFrame, scores: dict[str, pl.Expr]) -> pl.DataFrame:
    """Every possible shift of each day, in date order and, within a day, in the order
    `best_shift` scans them: `date`, `start` (the German hour the shift starts),
    `cost` (EUR of electricity plus premium for its 500 MWh at the cleared price) and,
    for each named score, its total over the shift's 8 hours."""
    position = pl.int_range(pl.len()).over("date")
    totals = {
        name: score.rolling_sum(SHIFT_HOURS).over("date")
        for name, score in scores.items()
    }
    return (
        frame.sort("utc_timestamp")
        .with_columns(
            start=pl.col("hour").shift(SHIFT_HOURS - 1).over("date"),
            cost=((pl.col("price_da") + premium()) * SHIFT_MW)
            .rolling_sum(SHIFT_HOURS)
            .over("date"),
            **totals,
        )
        .filter(position >= SHIFT_HOURS - 1)  # windows that end inside the day
        .select("date", "start", "cost", *scores)
    )


def pick_freely(windows: pl.DataFrame, score: str) -> pl.DataFrame:
    """Each day's window with the lowest `score`, the first one on a tie, as `best_shift`
    picks it. One row of `windows` per date."""
    is_lowest = pl.int_range(pl.len()).over("date") == pl.col(score).arg_min().over(
        "date"
    )
    return windows.filter(is_lowest)


def pick_with_rest(windows: pl.DataFrame, score: str) -> pl.DataFrame:
    """Each day's shift, chosen in date order the day before: the window with the lowest
    `score` among those starting at least 11 hours after the previous day's chosen
    shift ends. The first day is free. One row of `windows` per date."""
    picks: list[pl.DataFrame] = []
    previous_start: int | None = None
    for day in windows.partition_by("date", maintain_order=True):
        allowed = (
            day
            if previous_start is None
            else day.filter(pl.col("start") >= previous_start + EARLIEST_START_AFTER)
        )
        pick = allowed.pipe(pick_freely, score)
        previous_start = pick["start"].item()
        picks.append(pick)
    return pl.concat(picks)


def pick_weekly(windows: pl.DataFrame, score: str, keep_rest: bool) -> pl.DataFrame:
    """One start for each Monday-to-Sunday week: the start whose `score`, averaged over
    the week's days, is lowest, the earliest on a tie. The mean rather than the sum, so a
    start missing from a clock-change day is not favoured for having one window fewer.
    With `keep_rest`, the weeks are picked in date order and a start more than 5 hours
    earlier than the previous week's last start is not allowed, so the rest rule holds
    at the week boundary as well as inside the week. On a 23-hour day with no window at
    the week's start, the day takes the start with the next lowest weekly mean. The
    25-hour October day has two windows starting at 02:00, and both count in that
    start's mean. One row of `windows` per date."""
    with_totals = windows.with_columns(
        week=pl.col("date") - pl.duration(days=pl.col("date").dt.weekday() - 1)
    ).with_columns(week_mean=pl.col(score).mean().over("week", "start"))
    if not keep_rest:
        picks = with_totals.pipe(pick_freely, "week_mean")
        return picks.drop("week", "week_mean")
    weeks: list[pl.DataFrame] = []
    previous_start: int | None = None
    for week in with_totals.partition_by("week", maintain_order=True):
        allowed = (
            week
            if previous_start is None
            else week.filter(pl.col("start") >= previous_start + EARLIEST_START_AFTER)
        )
        picks = allowed.pipe(pick_freely, "week_mean")
        previous_start = picks["start"][-1]
        weeks.append(picks)
    return pl.concat(weeks).drop("week", "week_mean")


def cheapest_with_rest(windows: pl.DataFrame) -> pl.DataFrame:
    """The sequence of shifts with the lowest total `cost` over the whole period in which
    every shift starts at least 11 hours after the one before ends: hindsight under the
    rest rule, found by dynamic programming over the days. One row of `windows` per
    date."""
    days = windows.partition_by("date", maintain_order=True)
    # best[d][w]: the lowest total cost from the first day to day d that ends on window w
    # of day d; came_from[d][w]: the window of day d-1 that total runs through
    best: list[list[float]] = [days[0]["cost"].to_list()]
    came_from: list[list[int]] = [[-1] * days[0].height]
    for d in range(1, len(days)):
        previous_starts = days[d - 1]["start"].to_list()
        best_today: list[float] = []
        from_today: list[int] = []
        for start, cost in zip(days[d]["start"], days[d]["cost"], strict=True):
            allowed = [
                k
                for k, previous_start in enumerate(previous_starts)
                if start >= previous_start + EARLIEST_START_AFTER
            ]
            k = min(allowed, key=lambda k: best[d - 1][k])
            best_today.append(best[d - 1][k] + cost)
            from_today.append(k)
        best.append(best_today)
        came_from.append(from_today)
    # walk back from the cheapest last window
    w = min(range(len(best[-1])), key=lambda w: best[-1][w])
    chosen: list[int] = []
    for d in range(len(days) - 1, -1, -1):
        chosen.append(w)
        w = came_from[d][w]
    chosen.reverse()
    return pl.concat([day[w] for day, w in zip(days, chosen, strict=True)])


def complete_shift_windows(
    frame: pl.DataFrame, scores: dict[str, pl.Expr]
) -> pl.DataFrame:
    """`shift_windows`, refusing a period with a missing price or score or a gap in the
    dates. The pickers skip a day without scores and chain the rest rule across a gap,
    which would score such a period wrongly."""
    windows = frame.pipe(shift_windows, scores)
    has_missing = pl.any_horizontal(pl.col("cost", *scores).is_null()).any()
    if windows.select(has_missing).item():
        raise ValueError("a day has missing prices or scores")
    dates = windows["date"].unique().sort()
    if (dates.diff().drop_nulls().dt.total_days() != 1).any():
        raise ValueError("the period needs consecutive days")
    return windows


def count_rest_breaks(picks: pl.DataFrame) -> int:
    """The number of days whose shift starts less than 11 hours after the previous
    calendar day's shift ends. `picks` holds one row per date with `date` and `start`."""
    ordered = picks.sort("date")
    follows_previous_day = pl.col("date").diff().dt.total_days() == 1
    too_early = pl.col("start").diff() < EARLIEST_START_AFTER
    return ordered.filter(follows_previous_day & too_early).height


def count_start_changes(picks: pl.DataFrame) -> int:
    """The number of days whose shift starts at a different hour from the previous
    day's. `picks` holds one row per date with `date` and `start`, with no gap in the
    dates."""
    changed = picks.sort("date")["start"].diff() != 0
    return int(changed.sum())


def rest_check(
    frame: pl.DataFrame, scores: dict[str, pl.Expr], weekly: Collection[str] = ()
) -> pl.DataFrame:
    """For each named score and for hindsight, the rest breaks of its free schedule and
    what the rest rule costs. The scores named in `weekly` pick one start a week with
    `pick_weekly`, the rest one start a day. One row per schedule:

    - `rest_breaks`, `rest_break_share`: days that start too soon after the day before,
      as a count and as a share of all days.
    - `start_changes`, `start_changes_rest`: days whose start differs from the day
      before's, picked freely and under the rest rule.
    - `saving_share`, `saving_share_rest`: `(standard - picked) / (standard - hindsight)`
      from the period's total costs, picked freely, then picked under the rest rule
      against hindsight under the rest rule.
    - `saving_eur_per_year`: the free schedule's saving against the standard 06:00
      shift, over a year; `rest_eur_per_year`: the change in that saving under the rest
      rule, negative when the rule costs money.

    Each score picks the lowest total over 8 hours, so add the premium to a price
    forecast before passing it."""
    windows = frame.pipe(complete_shift_windows, scores)
    days = windows["date"].n_unique()
    free_picks = {
        name: windows.pipe(pick_weekly, name, False)
        if name in weekly
        else windows.pipe(pick_freely, name)
        for name in scores
    }
    rest_picks = {
        name: windows.pipe(pick_weekly, name, True)
        if name in weekly
        else windows.pipe(pick_with_rest, name)
        for name in scores
    }
    free_picks["hindsight"] = windows.pipe(pick_freely, "cost")
    rest_picks["hindsight"] = windows.pipe(cheapest_with_rest)
    # each schedule's total cost over the period in EUR. Series.sum() is typed as any
    # number, Decimal included, and a cost column is always a float
    free_cost = {name: float(picks["cost"].sum()) for name, picks in free_picks.items()}
    rest_cost = {name: float(picks["cost"].sum()) for name, picks in rest_picks.items()}
    standard = float(windows.filter(pl.col("start") == STANDARD_SHIFT[0])["cost"].sum())
    rows = [
        {
            "schedule": name,
            "days": days,
            "rest_breaks": count_rest_breaks(free_picks[name]),
            "rest_break_share": count_rest_breaks(free_picks[name]) / days,
            "start_changes": count_start_changes(free_picks[name]),
            "start_changes_rest": count_start_changes(rest_picks[name]),
            "saving_share": (standard - free_cost[name])
            / (standard - free_cost["hindsight"]),
            "saving_share_rest": (standard - rest_cost[name])
            / (standard - rest_cost["hindsight"]),
            "saving_eur_per_year": (standard - free_cost[name]) / days * 365,
            "rest_eur_per_year": (free_cost[name] - rest_cost[name]) / days * 365,
        }
        for name in free_picks
    ]
    return pl.DataFrame(rows)


def pick_near_plan(
    windows: pl.DataFrame,
    score: str,
    plan: pl.DataFrame,
    earlier: int,
    later: int,
    keep_rest: bool,
) -> pl.DataFrame:
    """Each day's window with the lowest `score` among those starting from `earlier`
    hours before to `later` hours after that day's planned start. `plan` holds one row
    per date with `date` and `start`, as the pickers return it. With `keep_rest`, the
    days are picked in date order as `pick_with_rest` does, which always finds a start
    when the plan itself keeps the rest rule: the planned start plus `later` is never
    too early. One row of `windows` per date."""
    planned = plan.select("date", planned_start="start")
    in_reach = pl.col("start").is_between(
        pl.col("planned_start") - earlier, pl.col("planned_start") + later
    )
    near = windows.join(planned, on="date").filter(in_reach).drop("planned_start")
    if keep_rest:
        return near.pipe(pick_with_rest, score)
    return near.pipe(pick_freely, score)


def make_plan(
    windows: pl.DataFrame, score: str, keep_rest: bool, weekly_plan: bool = False
) -> pl.DataFrame:
    """The plan a plan-then-move schedule starts from: each day's lowest `score`, or one
    start a week with `weekly_plan`, picked by `pick_weekly`; with `keep_rest`, kept
    within the rest rule. One row of `windows` per date, for `pick_near_plan`."""
    if weekly_plan:
        return windows.pipe(pick_weekly, score, keep_rest)
    if keep_rest:
        return windows.pipe(pick_with_rest, score)
    return windows.pipe(pick_freely, score)


def schedule_summary(
    windows: pl.DataFrame, picks: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    """One row per named schedule, each given as its picks, one row of `windows` per
    date:

    - `rest_breaks`: days that start less than 11 hours after the day before's shift.
    - `start_changes`: days whose start differs from the day before's.
    - `saving_share`: `(standard - picked) / (standard - hindsight)` from the period's
      total costs, with the free daily hindsight as the denominator for every schedule,
      rest kept or not.
    - `saving_share_rest_hindsight`: the same against hindsight under the rest rule,
      the denominator `rest_check` and `flex_check` use for rest-keeping schedules.
    - `saving_eur_per_year`: the saving against the standard 06:00 shift, over a year.
    - `cost_eur_per_year`: the schedule's own electricity plus premium, over a year.
    """
    days = windows["date"].n_unique()
    standard = float(windows.filter(pl.col("start") == STANDARD_SHIFT[0])["cost"].sum())
    free_hindsight = float(windows.pipe(pick_freely, "cost")["cost"].sum())
    rest_hindsight = float(windows.pipe(cheapest_with_rest)["cost"].sum())
    rows: list[dict[str, str | int | float]] = []
    for name, chosen in picks.items():
        cost = float(chosen["cost"].sum())
        rows.append(
            {
                "schedule": name,
                "days": days,
                "rest_breaks": count_rest_breaks(chosen),
                "start_changes": count_start_changes(chosen),
                "saving_share": (standard - cost) / (standard - free_hindsight),
                "saving_share_rest_hindsight": (standard - cost)
                / (standard - rest_hindsight),
                "saving_eur_per_year": (standard - cost) / days * 365,
                "cost_eur_per_year": cost / days * 365,
            }
        )
    return pl.DataFrame(rows)


def flex_check(
    frame: pl.DataFrame,
    plan_score: pl.Expr,
    final_score: pl.Expr,
    reaches: dict[str, tuple[int, int]],
    weekly_plan: bool = False,
) -> pl.DataFrame:
    """Plan each day's shift with `plan_score`, then move it with `final_score` to the
    cheapest start within reach of the plan. With `weekly_plan`, the plan holds one
    start a week, picked by `pick_weekly`. `reaches` maps a name to how many hours
    the start may move earlier and later. Each reach is scored twice, choosing freely
    and keeping the rest rule, where the plan also keeps it. One row per reach and
    setting:

    - `reach`, `earlier`, `later`, `keep_rest`: the reach and the setting.
    - `saving_share`: `(standard - picked) / (standard - hindsight)` from the period's
      total costs, against free hindsight, or hindsight under the rest rule when
      `keep_rest`, as in `rest_check`.
    - `saving_eur_per_year`: the saving against the standard 06:00 shift, over a year.
    - `moved_share`, `mean_move_hours`: the share of days whose final start differs
      from the planned one, and the mean distance moved in hours.

    Add the premium to a price forecast before passing it, as for `rest_check`."""
    windows = frame.pipe(
        complete_shift_windows, {"plan": plan_score, "final": final_score}
    )
    days = windows["date"].n_unique()
    standard = float(windows.filter(pl.col("start") == STANDARD_SHIFT[0])["cost"].sum())
    hindsight = {
        False: float(windows.pipe(pick_freely, "cost")["cost"].sum()),
        True: float(windows.pipe(cheapest_with_rest)["cost"].sum()),
    }
    rows: list[dict[str, str | int | bool | float]] = []
    for keep_rest in (False, True):
        plan = windows.pipe(make_plan, "plan", keep_rest, weekly_plan)
        for name, (earlier, later) in reaches.items():
            picks = windows.pipe(
                pick_near_plan, "final", plan, earlier, later, keep_rest
            )
            cost = float(picks["cost"].sum())
            # both in date order, one row a day
            move = picks["start"].to_numpy() - plan["start"].to_numpy()
            rows.append(
                {
                    "reach": name,
                    "earlier": earlier,
                    "later": later,
                    "keep_rest": keep_rest,
                    "saving_share": (standard - cost)
                    / (standard - hindsight[keep_rest]),
                    "saving_eur_per_year": (standard - cost) / days * 365,
                    "moved_share": float(np.mean(move != 0)),
                    "mean_move_hours": float(np.abs(move).mean()),
                }
            )
    return pl.DataFrame(rows)
