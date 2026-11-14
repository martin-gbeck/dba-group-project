# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: spot-price-forecast (uv)
#     language: python
#     name: spot-price-forecast
# ---

# %% [markdown]
# # Choosing the shifts a week ahead
#
# **Contents**
#
# - Objective
# - 1\. Timing: what is known a week ahead
# - 2\. Data and features
# - 3\. The model
# - 4\. Scores
# - 5\. Rest between shifts
# - 6\. Plan the week, adjust the day before
# - What a week's notice costs
#
# ## Objective
#
# The main notebook, `de_price_forecast.ipynb`, picks each day's shift at 11:55 the day before. Here the plant fixes all seven shifts of the coming week at once, which is the notice a crew roster needs. The question is how much of the saving survives when the forecast has to reach up to a week ahead.
#
# Everything else stays as in the main notebook: one 8-hour shift a day starting between 00:00 and 16:00, 500 MWh, the same night and weekend premium, the same `saving_share` and the same years. Only the inputs change, because fewer of them are known a week early.
#

# %% [markdown]
# ## 1. Timing: what is known a week ahead
#
# The schedule for Monday to Sunday is fixed on the Sunday before, at 11:55, the same time of day as the main notebook's gate. By then Sunday's own prices are public, since its auction cleared on Saturday at noon. So the last known prices are the Sunday's, and the forecast reaches 1 day ahead for Monday and 7 days ahead for Sunday.
#
# That rules out three kinds of input:
#
# | Input in the main model | A week ahead |
# |---|---|
# | Same hour on D-1 and D-2 (lag 24 and 48) | Not known for Tuesday to Sunday. Replaced by the same hour on the decision Sunday |
# | D-1's mean, min and max, and the 7-day and 30-day means up to D-1 | Replaced by the same statistics up to the decision Sunday |
# | The TSO's day-ahead load, solar and wind forecasts | Not published that far out. Energy-Charts, the source of all data here, serves load only as a day-ahead forecast, and a week-ahead load forecast would need ENTSO-E's API, a key and a new fetcher, so it is left out |
#
# The calendar and the price at the same hour 7 days earlier stay legal for every day of the week: for the Sunday being scheduled, 7 days back is the decision Sunday itself. The week-ahead model therefore sees the clock and recent prices only, nothing about the weather.
#

# %% [markdown]
# Load the libraries and the fetch functions, and set the table display defaults.
#

# %%
import numpy as np
import polars as pl
from plotnine import (
    aes,
    facet_wrap,
    geom_hline,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_x_continuous,
    theme,
    theme_minimal,
    theme_set,
)

from lib.data_schemas import CombinedDataSchema
from lib.decision import flex_check, premium, rest_check
from lib.evaluation import Scores, cross_validate, evaluate, summarise_folds
from lib.fetching import fetch_de_prices, fetch_forecasts, fetch_neighbor_prices
from lib.modeling import LassoThenXGB, df_to_np
from lib.splitting import make_folds

# a kernel has no terminal, so polars would squeeze every table into ~100 characters
_ = (
    pl.Config.set_tbl_rows(30)
    .set_tbl_cols(-1)  # show every column
    .set_tbl_width_chars(1000)  # wide tables use the pane instead of wrapping
    .set_fmt_str_lengths(60)
    .set_float_precision(3)
)
theme_set(theme_minimal() + theme(figure_size=(10, 4.5), legend_position="top"))

# %% [markdown]
# ## 2. Data and features
#
# The hourly table is built exactly as in sections 1.2 and 4.1 of the main notebook: every file averaged to UTC hours, joined onto the prices, calendar columns in German time, validated against the same data model.
#

# %% [markdown]
# Load the three raw files, put them on one hourly grid, join them, and validate the result.
#


# %%
def to_hourly(raw: pl.DataFrame) -> pl.DataFrame:
    return (
        raw.sort("utc_timestamp")
        .group_by_dynamic("utc_timestamp", every="1h")
        .agg(pl.all().mean())
    )


local = pl.col("utc_timestamp").dt.convert_time_zone("Europe/Berlin")
merged = (
    to_hourly(fetch_de_prices())
    .join(to_hourly(fetch_forecasts()), on="utc_timestamp", how="left")
    .join(to_hourly(fetch_neighbor_prices()), on="utc_timestamp", how="left")
    .with_columns(
        date=local.dt.date(),
        year=local.dt.year(),
        hour=local.dt.hour(),
        weekday=local.dt.weekday(),
        fc_residual=pl.col("fc_load")
        - pl.sum_horizontal("fc_solar", "fc_wind_onshore", "fc_wind_offshore"),
    )
)
df = CombinedDataSchema.validate(merged).sort("utc_timestamp")
df.height

# %% [markdown]
# Validation passed, so the table is the main notebook's, hour for hour.
#

# %% [markdown]
# Two feature sets are built from it. The day-ahead set is the final model's 17 columns from the main notebook, rebuilt here so both models are scored side by side. The week-ahead set swaps every price input for its week-ahead version and has no forecasts:
#
# - `decision_date`: the Sunday the week's schedule is fixed on, `date - weekday` days, since Monday is weekday 1 and Sunday weekday 7.
# - `price_decision_day`: the price at the same hour on `decision_date`. For Monday it is the same as lag 24, for Sunday the same as lag 168.
# - `price_lag7d`, `price_lag14d`: the same hour 7 and 14 days earlier.
#
# The week-ahead prices are looked up by German date and hour rather than by shifting a fixed number of hours, as the main notebook's lags do. A shift of 168 hours reads one hour too far forward on the Sunday after the October clock change, where 23:00 lands on Monday 00:00, a price not yet public on the decision Sunday.
# - `decision_day_mean`, `decision_day_min`, `decision_day_max`, `decision_mean_7d`, `decision_mean_30d`: the decision Sunday's mean, min and max price, and the mean daily price over the 7 and 30 days up to and including it.
#
# These mirror the main model's eight price-memory columns one for one, which keeps the two models as alike as the timing allows. `price_lag24` is kept in the table for one job only: MASE is measured against same hour yesterday in both notebooks.
#

# %% [markdown]
# Compute each day's price statistics, then attach them to the next day for the day-ahead set and to the coming week for the week-ahead set.
#

# %%
daily = (
    df.group_by("date")
    .agg(
        day_mean=pl.col("price_da").mean(),
        day_min=pl.col("price_da").min(),
        day_max=pl.col("price_da").max(),
    )
    .sort("date")
    .with_columns(
        mean_7d=pl.col("day_mean").rolling_mean(7),
        mean_30d=pl.col("day_mean").rolling_mean(30),
    )
)
# the day-ahead set sees the statistics up to D-1, as in the main notebook
day_ahead_stats = daily.select(
    (pl.col("date") + pl.duration(days=1)).alias("date"),
    prev_day_mean="day_mean",
    prev_day_min="day_min",
    prev_day_max="day_max",
    mean_7d="mean_7d",
    mean_30d="mean_30d",
)
# the week-ahead set sees them up to the decision Sunday
week_ahead_stats = daily.select(
    pl.col("date").alias("decision_date"),
    decision_day_mean="day_mean",
    decision_day_min="day_min",
    decision_day_max="day_max",
    decision_mean_7d="mean_7d",
    decision_mean_30d="mean_30d",
)
# one price per German date and hour, for the week-ahead lookups below; the 25-hour day
# in October has hour 2 twice, and the two are averaged
local_prices = df.group_by("date", "hour").agg(price=pl.col("price_da").mean())

features = (
    df.with_columns(
        # the grid is hourly with no gaps, so shifting by rows is shifting by hours
        price_lag24=pl.col("price_da").shift(24),
        price_lag48=pl.col("price_da").shift(48),
        price_lag168=pl.col("price_da").shift(168),
        month=pl.col("date").dt.month(),
        is_weekend=pl.col("weekday") >= 6,
        decision_date=pl.col("date") - pl.duration(days=pl.col("weekday")),
    )
    .with_columns(
        # the day each week-ahead price is read from
        price_decision_day_source=pl.col("decision_date"),
        price_lag7d_source=pl.col("date") - pl.duration(days=7),
        price_lag14d_source=pl.col("date") - pl.duration(days=14),
    )
    .join(day_ahead_stats, on="date", how="left")
    .join(week_ahead_stats, on="decision_date", how="left")
)
for price in ("price_decision_day", "price_lag7d", "price_lag14d"):
    features = features.join(
        local_prices.rename({"date": f"{price}_source", "price": price}),
        on=[f"{price}_source", "hour"],
        how="left",
    )
features = features.sort("utc_timestamp").with_columns(
    # the March clock change skips hour 2, so a lookup into that day finds no price for
    # hour 2 and takes the hour before it, which is earlier and so still known
    pl.col("price_decision_day", "price_lag7d", "price_lag14d").forward_fill()
)

# %% [markdown]
# Write out both column lists, and the columns each Lasso stage sees. As in the main notebook, the day-ahead Lasso gets residual load and solar in place of the five forecasts. The week-ahead model has no forecasts, so its Lasso sees every column.
#

# %%
CALENDAR = ["hour", "weekday", "month", "is_weekend"]
DAY_AHEAD_COLUMNS = [
    *CALENDAR,
    "fc_load",
    "fc_solar",
    "fc_wind_onshore",
    "fc_wind_offshore",
    "fc_residual",
    "price_lag24",
    "price_lag48",
    "price_lag168",
    "prev_day_mean",
    "prev_day_min",
    "prev_day_max",
    "mean_7d",
    "mean_30d",
]
DAY_AHEAD_LINEAR = [
    column
    for column in DAY_AHEAD_COLUMNS
    if column not in ("fc_load", "fc_wind_onshore", "fc_wind_offshore")
]
WEEK_AHEAD_COLUMNS = [
    *CALENDAR,
    "price_decision_day",
    "price_lag7d",
    "price_lag14d",
    "decision_day_mean",
    "decision_day_min",
    "decision_day_max",
    "decision_mean_7d",
    "decision_mean_30d",
]

# %% [markdown]
# Check that every week-ahead price column is known at the decision time. `latest_day_used` is the last delivery day a column reads, counted from the decision Sunday: 0 means the Sunday itself, a positive number would mean a price that is not yet public. The check runs over every hour of every week in the data.
#

# %%
pl.DataFrame(
    [
        {
            "column": column,
            "latest_day_used": features.select(
                (pl.col(source) - pl.col("decision_date")).dt.total_days().max()
            ).item(),
        }
        for column, source in [
            ("price_decision_day", "price_decision_day_source"),
            ("price_lag7d", "price_lag7d_source"),
            ("price_lag14d", "price_lag14d_source"),
            ("decision_day_mean, min, max, mean_7d, mean_30d", "decision_date"),
        ]
    ]
)

# %% [markdown]
# No column reads past the decision Sunday, so the whole week-ahead set is known when the week's shifts are fixed.
#

# %% [markdown]
# Split the table into the development years and the 2025 test year. Each model drops only the first days of 2019 that its own columns cannot fill, so the day-ahead rows are exactly the main notebook's `dev`. The 2026 reserve is not used here.
#

# %%
TEST_YEAR = 2025
FOLDS = {"fold-1": 2023, "fold-2": 2024}  # fold name: validation year

before_test = features.filter(pl.col("year") < TEST_YEAR)
dev_day_ahead = before_test.drop_nulls(DAY_AHEAD_COLUMNS)
dev_week_ahead = before_test.drop_nulls(WEEK_AHEAD_COLUMNS)
test_2025 = features.filter(pl.col("year") == TEST_YEAR)

parts: list[tuple[str, pl.DataFrame, list[str]]] = [
    ("dev-day-ahead", dev_day_ahead, DAY_AHEAD_COLUMNS),
    ("dev-week-ahead", dev_week_ahead, WEEK_AHEAD_COLUMNS),
    (
        "test-2025",
        test_2025,
        list(dict.fromkeys(DAY_AHEAD_COLUMNS + WEEK_AHEAD_COLUMNS)),
    ),
]
pl.DataFrame(
    [
        {
            "set": name,
            "hours": part.height,
            "first_day": part["date"].min(),
            "last_day": part["date"].max(),
            "missing_feature_values": part.select(columns)
            .null_count()
            .sum_horizontal()
            .item(),
        }
        for name, part, columns in parts
    ]
)

# %% [markdown]
# The week-ahead training data starts a week later, on 2019-02-07, because its 30-day mean has to end on the Sunday before. Neither model has a missing value in 2025.
#

# %% [markdown]
# ## 3. The model
#
# Both rows use the main notebook's final model with its settings unchanged: Lasso-then-XGBoost, a penalty of 3, 500 trees 6 levels deep, a learning rate of 0.03. Nothing is tuned for the week-ahead inputs. The question is what the shorter view costs, and a new search now, with 2025 already scored, would be choosing on the test year again.
#
# The week-ahead model is scored on 2025 as well. It does not compete with the day-ahead model for anything: the final model in the main notebook stays final whatever this one scores, so the 2025 number cannot feed back into a choice.
#

# %% [markdown]
# Build both models with the final settings.
#

# %%
FINAL_SETTINGS = {
    "alpha": 3.0,
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 500,
}
models = {
    "day-ahead": (
        LassoThenXGB(DAY_AHEAD_COLUMNS, DAY_AHEAD_LINEAR, **FINAL_SETTINGS),
        DAY_AHEAD_COLUMNS,
        dev_day_ahead,
    ),
    "week-ahead": (
        LassoThenXGB(WEEK_AHEAD_COLUMNS, WEEK_AHEAD_COLUMNS, **FINAL_SETTINGS),
        WEEK_AHEAD_COLUMNS,
        dev_week_ahead,
    ),
}

# %% [markdown]
# ## 4. Scores
#
# The metrics are the main notebook's, from `evaluate` in `lib/evaluation.py`:
#
# - `saving_share`: `(standard - shift_cost) / (standard - hindsight)`, each a year's average cost in EUR/MWh, electricity plus premium. 100% matches hindsight, 0% is the standard 06:00 to 14:00 shift.
# - `saving_eur_per_year`: what the picked shifts save against the standard shift over a year, for 500 MWh a day.
# - `mae` in EUR/MWh, and `mase`, the MAE over that of same hour yesterday, for the course.
#
# Two references bound the result. `last-year-profile` picks every day's shift from the previous year's average price by hour, so it could be fixed a year ahead: it is the floor for any schedule set in advance. `standard-shift` scores 0 by definition. `cross_validate` fits each model on the years before each validation year, as in the main notebook, and the validation columns are the mean of 2023 and 2024.
#

# %% [markdown]
# Score both models on the validation years, then refit each on all development years and score 2025.
#

# %%
validation: dict[str, Scores] = {}
test_scores: dict[str, Scores] = {}
validation_forecasts: dict[str, pl.DataFrame] = {}  # out-of-fold, 2023 and 2024
test_forecasts: dict[str, np.ndarray] = {}
for name, (model, columns, dev) in models.items():
    validation[name], validation_forecasts[name] = cross_validate(
        model, columns, dev, FOLDS
    )
    _ = model.fit(df_to_np(dev, columns), dev["price_da"].to_numpy())
    test_forecasts[name] = model.predict(df_to_np(test_2025, columns))
    test_scores[name] = evaluate(test_2025, test_forecasts[name])

# %% [markdown]
# Score the two references the same way: on each validation year, then on 2025, each year's profile taken from the year before it.
#


# %%
def reference_scores(fit: pl.DataFrame, valid: pl.DataFrame) -> dict[str, Scores]:
    last_year = (
        fit.filter(pl.col("year") == fit["year"].max())
        .group_by("hour")
        .agg(profile=pl.col("price_da").mean())
    )
    profile = valid.join(last_year, on="hour", how="left").sort("utc_timestamp")[
        "profile"
    ]
    standard = np.where(valid["hour"].is_between(6, 13).to_numpy(), 0.0, 1.0)
    return {
        "last-year-profile": evaluate(valid, profile.to_numpy(), report_error=False),
        "standard-shift": evaluate(
            valid, standard, pick_with_premium=False, report_error=False
        ),
    }


per_fold = [
    reference_scores(fit, valid)
    for fit, valid in make_folds(dev_day_ahead, FOLDS).values()
]
for name in ("last-year-profile", "standard-shift"):
    validation[name] = summarise_folds([fold[name] for fold in per_fold])
test_scores |= reference_scores(dev_day_ahead, test_2025)

# %% [markdown]
# Line the four up: validation mean first, then 2025.
#

# %%
pl.DataFrame(
    [
        {
            "method": name,
            "saving_share_validation": validation[name]["saving_share"],
            "saving_share_2025": test_scores[name]["saving_share"],
            "saving_eur_per_year_validation": validation[name]["saving_eur_per_year"],
            "saving_eur_per_year_2025": test_scores[name]["saving_eur_per_year"],
            "mae_validation": validation[name]["mae"],
            "mae_2025": test_scores[name]["mae"],
            "mase_2025": test_scores[name]["mase"],
        }
        for name in ("day-ahead", "week-ahead", "last-year-profile", "standard-shift")
    ]
)

# %% [markdown]
# A week ahead, the model gets 86.1% of the possible saving on the validation years and 90.7% on 2025, against 96.7% and 95.9% for the day-ahead model. On 2025 that is 4.05 million EUR a year instead of 4.29 million: fixing the week early costs about 230,000 EUR a year, and it still saves a million more than last year's profile.
#
# Its price forecast is worse than same hour yesterday (MASE 1.03), yet it picks the shift well. The level is hard to see a week out, but the shift only depends on the shape of the day, and the clock and last week's prices carry most of that. Year by year it gets 82.8% in 2023, 89.4% in 2024 and 90.7% in 2025: worst in 2023, the first year after the gas crisis, when the level moved most.
#

# %% [markdown]
# ## 5. Rest between shifts
#
# German law (Arbeitszeitgesetz §5) requires 11 hours off between two shifts, so a shift may start at most 5 hours earlier than the day before's. Section 8.4 of the main notebook counts how often the day-ahead schedules break this and what keeping it costs; this section adds the week-ahead schedule, with the same columns:
#
# - `rest_breaks`, `rest_break_share`: days in 2025 whose shift starts less than 11 hours after the day before's ends, as a count and as a share of the year's days. `rest_breaks_validation`, `rest_break_share_validation`: the same over 2023 and 2024 together, from the out-of-fold forecasts.
# - `saving_share`: from the free choice, as in section 4.
# - `saving_share_rest`: the same share with the rule kept. Each day takes its cheapest forecast shift among those that start at least 11 hours after the previous day's shift ends. A week-ahead plan is made in one go on the Sunday, but Sunday's own shift was fixed the week before, so day by day in date order gives the same constraint. Hindsight under the rule is the cheapest sequence over the whole year that never breaks it, found by dynamic programming.
# - `saving_eur_per_year`: the free choice's saving against the standard 06:00 shift. `rest_eur_per_year`: how that saving changes when the rule is kept, negative when the rule costs money.
#
# The rule compares clock hours, so on the two clock-change days a year the real rest is one hour shorter (March) or longer (October) than counted.
#

# %% [markdown]
# Count the rest breaks of each schedule on 2025 and on the validation years, and score each one with the rule kept.
#

# %%
test_rest_frame = test_2025.with_columns(
    day_ahead=pl.Series(test_forecasts["day-ahead"]),
    week_ahead=pl.Series(test_forecasts["week-ahead"]),
)
REST_SCORES = {
    "day-ahead": pl.col("day_ahead") + premium(),
    "week-ahead": pl.col("week_ahead") + premium(),
    "residual-load-rule": pl.col("fc_residual"),
}
rest_2025 = test_rest_frame.pipe(rest_check, REST_SCORES)

# the two models' validation rows differ in their first days of 2019, never in 2023 or
# 2024, so the forecasts line up on the timestamp
validation_rest_frame = (
    validation_forecasts["day-ahead"]
    .rename({"pred": "day_ahead"})
    .join(
        validation_forecasts["week-ahead"].select("utc_timestamp", week_ahead="pred"),
        on="utc_timestamp",
    )
)
# each validation year on its own, so no schedule runs across the gap between folds
rest_validation = (
    pl.concat(
        [
            validation_rest_frame.filter(pl.col("year") == year).pipe(
                rest_check, REST_SCORES
            )
            for year in FOLDS.values()
        ]
    )
    .group_by("schedule", maintain_order=True)
    .agg(
        rest_breaks_validation=pl.col("rest_breaks").sum(),
        rest_break_share_validation=pl.col("rest_breaks").sum() / pl.col("days").sum(),
    )
)
rest_2025.join(rest_validation, on="schedule").select(
    "schedule",
    "rest_breaks",
    "rest_break_share",
    "rest_breaks_validation",
    "rest_break_share_validation",
    "saving_share",
    "saving_share_rest",
    "saving_eur_per_year",
    "rest_eur_per_year",
)

# %% [markdown]
# The week-ahead schedule breaks the rest rule on 12 days of 2025 (3.3%) and on 40 validation days (5.5%), about half as often as the day-ahead model. Keeping the rule costs it 129,000 EUR a year, against 279,000 for the day-ahead model. With rest kept on both, choosing each day the day before saves about 83,000 EUR a year more than fixing the week on Sunday, instead of about 230,000.
#

# %% [markdown]
# ## 6. Plan the week, adjust the day before
#
# Crews need their shifts planned well ahead, and the day-ahead forecast picks the better shift. This section combines the two. Each day's shift is planned on the Sunday before, as in section 4, and at 11:55 the day before, the day-ahead model may move that day's start a few hours from the plan. The crew then knows its shift a week ahead, and it moves by a few hours at most. The plan rests on the weaker week-ahead forecast, which has no weather input: no TSO publishes wind or solar forecasts a week ahead, and the week-ahead load forecast is not in this data (section 1), but it is still a forecast of the coming week's prices, not a fixed timetable.
#
# - Planned start: the week-ahead model's cheapest forecast shift, the schedule of section 4.
# - Final start: the day-ahead model's cheapest forecast shift among the starts from `earlier` hours before to `later` hours after the planned one, still between 00:00 and 16:00. `pick_near_plan` in `lib/decision.py` makes this choice and `flex_check` scores it.
# - Reach: either way, `either-way-1h` to `either-way-4h`, where the start may move up to W hours earlier or later; or earlier only, `earlier-only-1h` to `earlier-only-4h`, where the plan is the latest a shift can start. `either-way-0h` keeps the plan, so it is the week-ahead schedule. `any-start` allows 16 hours, which covers every start, so it is the day-ahead choice of the main notebook.
# - With the rest rule kept, the plan is the week-ahead schedule with rest kept from section 5, and the final start must also come at least 11 hours after the previous day's final shift ends. A start that works always exists: the plan keeps the rule, so the planned start plus the later reach is never too early.
# - `saving_share`: `(standard - shift_cost) / (standard - hindsight)` from the year's total costs. With the rule kept, hindsight is the cheapest rest-keeping sequence, as in section 5. `saving_eur_per_year`: the saving against the standard 06:00 shift over a year.
# - `moved_share`: the share of 2025's days whose final start differs from the planned one. `mean_move_hours`: the mean distance moved, in hours, over all days.
#
# The validation columns are the mean of 2023 and 2024, each year scored on its own.
#

# %% [markdown]
# Score every reach on 2025 and on each validation year, with the free choice and with the rest rule kept.
#

# %%
REACH_HOURS = [0, 1, 2, 3, 4]
# name: (hours the start may move earlier, hours it may move later)
REACHES: dict[str, tuple[int, int]] = {
    f"either-way-{hours}h": (hours, hours) for hours in REACH_HOURS
} | {f"earlier-only-{hours}h": (hours, 0) for hours in REACH_HOURS if hours > 0}
# every start from 00:00 to 16:00 is within 16 hours, so this is the day-ahead choice
REACHES["any-start"] = (16, 16)
plan_score = REST_SCORES["week-ahead"]
final_score = REST_SCORES["day-ahead"]

flex_2025 = test_rest_frame.pipe(flex_check, plan_score, final_score, REACHES)
# each validation year on its own, then the mean of the two, as in section 4
flex_validation = (
    pl.concat(
        [
            validation_rest_frame.filter(pl.col("year") == year).pipe(
                flex_check, plan_score, final_score, REACHES
            )
            for year in FOLDS.values()
        ]
    )
    .group_by("reach", "keep_rest", maintain_order=True)
    .agg(
        saving_share_validation=pl.col("saving_share").mean(),
        saving_eur_per_year_validation=pl.col("saving_eur_per_year").mean(),
    )
)
flex = flex_2025.join(flex_validation, on=["reach", "keep_rest"]).select(
    "reach",
    "keep_rest",
    "saving_share_validation",
    "saving_share",
    "saving_eur_per_year_validation",
    "saving_eur_per_year",
    "moved_share",
    "mean_move_hours",
)
flex

# %% [markdown]
# Choosing freely, a reach of one hour either way lifts the 2025 saving share from 90.7% to 93.1%, or 4.16 million EUR a year: about 105,000 of the 233,000 EUR a year that a week's notice costs. Wider reaches add little, 93.8% at four hours, and the start moves on about 35% of days, by 0.35 to 0.54 hours on average. On the validation years the gain is smaller, from 86.1% to 87.9% at one hour and 88.9% at four, about 74,000 and 118,000 of a 438,000 gap. Moving earlier only wins back roughly half as much as moving either way.
#
# With the rest rule kept, every either-way reach of one hour or more saves more than choosing each day the day before: 4.04 million EUR a year at one hour and 4.08 million at four, against 4.01 million on 2025, and 3.56 and 3.61 million against 3.29 million on the validation years. The likely reason is the one in section 8.4 of the main notebook. Choosing one day at a time under the rule, a late shift today can rule out a cheap early shift tomorrow; a plan that already keeps the rule holds each day's choice near a sequence that works.
#

# %% [markdown]
# Plot the 2025 saving share against the reach, one panel for each setting, with the pure week-ahead and day-ahead levels as reference lines.
#

# %%
SETTING_NAMES = {False: "free choice", True: "11 hours of rest kept"}
# an ordered type, so the free choice is the first panel
SETTING_ORDER = pl.Enum(list(SETTING_NAMES.values()))
within_four = (pl.col("earlier") <= 4) & (pl.col("later") <= 4)
# the 0-hour reach starts both lines
either_way = flex_2025.filter(
    within_four, pl.col("earlier") == pl.col("later")
).with_columns(move=pl.lit("either way"), hours=pl.col("earlier"))
earlier_only = flex_2025.filter(within_four, pl.col("later") == 0).with_columns(
    move=pl.lit("earlier only"), hours=pl.col("earlier")
)
lines = pl.concat([either_way, earlier_only]).with_columns(
    setting=pl.col("keep_rest").replace_strict(
        SETTING_NAMES, return_dtype=SETTING_ORDER
    )
)
line_names = lines.filter(pl.col("hours") == 4)
references = pl.concat(
    [
        flex_2025.filter(pl.col("reach") == "any-start").with_columns(
            level=pl.lit("day-ahead choice")
        ),
        flex_2025.filter(pl.col("reach") == "either-way-0h").with_columns(
            level=pl.lit("week-ahead plan")
        ),
    ]
).with_columns(
    setting=pl.col("keep_rest").replace_strict(
        SETTING_NAMES, return_dtype=SETTING_ORDER
    )
)

(
    ggplot(lines, aes("hours", "saving_share"))
    + geom_hline(
        data=references,
        mapping=aes(yintercept="saving_share"),
        linetype="dashed",
        color="grey",
    )
    + geom_text(
        data=references,
        mapping=aes(x=2.5, y="saving_share", label="level"),
        va="bottom",
        size=8,
        color="grey",
    )
    + geom_line(aes(color="move"))
    + geom_point(aes(color="move"))
    + geom_text(
        data=line_names,
        mapping=aes(x=4.1, label="move", color="move"),
        ha="left",
        size=8,
    )
    + facet_wrap("setting")
    + scale_x_continuous(breaks=REACH_HOURS, limits=(0, 5.2))
    + labs(
        x="hours the day-ahead choice may move the planned start",
        y="share of the possible saving, 2025",
        title="A one-hour move wins back half of what a week's notice costs, and all of it with rest kept",
    )
    + theme(legend_position="none")
)

# %% [markdown]
# ## What a week's notice costs
#
# Fixing all seven shifts on the Sunday before keeps most of the saving, and letting the day before move each start by an hour wins back part of the rest.
#
# 1. On 2025 the week-ahead schedule gets 90.7% of the possible saving, 4.05 million EUR a year against the standard shift. Choosing each day the day before gets 4.29 million, so a week's notice costs about 230,000 EUR a year on 2025, and about 440,000 on the validation years.
# 2. Last year's profile, a schedule that needs no forecast at all, saves 3.05 million on 2025. The week-ahead model closes 81% of the gap between that floor and the day-ahead model on 2025, and 74% on the validation years.
# 3. What it gives up is the weather. No wind or solar forecast in this data reaches a week ahead, and a week-ahead weather forecast is the input most likely to win back part of the 230,000.
# 4. The model was not retuned for the longer horizon, so these figures are a cautious estimate of what a week-ahead schedule can do.
# 5. With the legal 11 hours of rest between shifts kept, a week's notice costs about 83,000 EUR a year instead of 230,000 (section 5). The day-ahead schedule breaks that rule on 7.4% of 2025's days and the week-ahead one on 3.3%, and keeping it costs the day-ahead schedule more.
# 6. Planning the week on Sunday and letting the day-ahead forecast move each start by up to an hour either way keeps the notice and wins back about 105,000 of the 233,000 EUR a year on 2025, though only 74,000 of 438,000 on the validation years (section 6). The start moves on about one day in three. With the rest rule kept, the same plan with a one-hour reach saves about 36,000 EUR a year more on 2025 than choosing each day the day before, so a week's notice costs nothing once the law is kept.
#
