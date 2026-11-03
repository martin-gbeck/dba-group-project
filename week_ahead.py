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

from lib.data_schemas import CombinedDataSchema
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
for name, (model, columns, dev) in models.items():
    validation[name], _ = cross_validate(model, columns, dev, FOLDS)
    _ = model.fit(df_to_np(dev, columns), dev["price_da"].to_numpy())
    test_scores[name] = evaluate(test_2025, model.predict(df_to_np(test_2025, columns)))

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
# ## What a week's notice costs
#
# Fixing all seven shifts on the Sunday before keeps most of the saving.
#
# 1. On 2025 the week-ahead schedule gets 90.7% of the possible saving, 4.05 million EUR a year against the standard shift. Choosing each day the day before gets 4.29 million, so a week's notice costs about 230,000 EUR a year on 2025, and about 440,000 on the validation years.
# 2. Last year's profile, a schedule that needs no forecast at all, saves 3.05 million on 2025. The week-ahead model closes 81% of the gap between that floor and the day-ahead model on 2025, and 74% on the validation years.
# 3. What it gives up is the weather. No wind or solar forecast in this data reaches a week ahead, and a week-ahead weather forecast is the input most likely to win back part of the 230,000.
# 4. The model was not retuned for the longer horizon, so these figures are a cautious estimate of what a week-ahead schedule can do.
#
