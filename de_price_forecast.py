# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # German day-ahead price forecast
#
# ## Objective
#
# A German manufacturer needs 500 MWh tomorrow and can run in any hours. At 11:55 on D-1, five minutes before the day-ahead auction closes, it picks the cheapest hours of day D. This notebook builds the price forecast it picks them from. The full case is in `project.md`.
#
# | | |
# |---|---|
# | Target | `price_da`, the DE-LU day-ahead clearing price in EUR/MWh, one row per delivery hour |
# | Predict or explain | Predict. The model is judged on held-out error, and its coefficients are a by-product |
# | Forecast moment | 11:55 German time on D-1, 12 to 36 hours ahead. Nothing published after that may be a feature |
# | What an error costs | Only the ranking of the hours inside day D changes the purchase. A forecast that is 20 EUR/MWh too high in every hour of a day costs the company nothing |
# | Metrics | MAE and RMSE for the course, MASE against same-hour-yesterday so years at different price levels compare, and the euros the schedule saves against a fixed schedule and against perfect hindsight |

# %% [markdown]
# Load the libraries and the three fetch functions, and set the plot and table display defaults.

# %%
import polars as pl
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_col,
    geom_hline,
    geom_line,
    geom_point,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient,
    scale_x_continuous,
    theme,
    theme_minimal,
    theme_set,
)

from data_schemas import CombinedDataSchema
from spot_price_forecast import fetch_de_prices, fetch_forecasts, fetch_neighbor_prices

theme_set(theme_minimal() + theme(figure_size=(10, 4.5)))
_ = pl.Config.set_tbl_rows(30).set_tbl_cols(25)

# %% [markdown]
# ## 1. Raw data
#
# `refresh=True` on any of the three pulls it again from the Energy-Charts API. The files hold
# what the API served, untouched.

# %% [markdown]
# Load the three raw datasets from `data/`, pulling from the API only if a file is missing.

# %%
prices_raw = fetch_de_prices()
forecasts_raw = fetch_forecasts()
neighbors_raw = fetch_neighbor_prices()

# %% [markdown]
# ## 2. Parse and join
#
# Mechanical fixes only, the same for every file and none of them learned from the data:
#
# 1. Every file is averaged to UTC hours. Prices and neighbour prices are hourly until the
#    market-time-unit change on 2025-10-01 and quarter-hourly after; the forecasts are
#    quarter-hourly throughout. The hourly mean is the hour's average price, and for the
#    forecasts the hour's average power. It also repairs Switzerland, which still publishes hourly:
#    three of its four quarter-hour rows are empty, and the mean skips them.
# 2. The files are joined on `utc_timestamp`, starting from the price table, so every row has a target.
#    Forecast hours past the last cleared auction drop out here.
# 3. German local time and the calendar columns come from the UTC timestamp. Hour of day is
#    local because demand follows the clock people live by.
# 4. `fc_residual` is load minus the three renewable forecasts, as defined in `project.md`. It is
#    arithmetic on one row and safe before the split.
# 5. The merged table is validated against the data model in `data_schemas.py`. It casts every
#    column to its type, drops any column the model does not list, and checks the hourly grid,
#    the price limits and which columns may be empty. Open that file to see what the table holds.

# %% [markdown]
# Average every file to UTC hours.


# %%
def to_hourly(raw: pl.DataFrame) -> pl.DataFrame:
    return (
        raw.sort("utc_timestamp")
        .group_by_dynamic("utc_timestamp", every="1h")
        .agg(pl.all().mean())
    )


prices = to_hourly(prices_raw)
forecasts = to_hourly(forecasts_raw)
neighbors = to_hourly(neighbors_raw)

# %% [markdown]
# Join the forecasts and neighbour prices onto the price table by timestamp.

# %%
joined = prices.join(forecasts, on="utc_timestamp", how="left").join(
    neighbors, on="utc_timestamp", how="left"
)

# %% [markdown]
# Add the German date, year, hour of day and weekday, and residual load.

# %%
local = pl.col("utc_timestamp").dt.convert_time_zone("Europe/Berlin")

merged = joined.with_columns(
    date=local.dt.date(),
    year=local.dt.year(),
    hour=local.dt.hour(),
    weekday=local.dt.weekday(),
    fc_residual=pl.col("fc_load")
    - pl.sum_horizontal("fc_solar", "fc_wind_onshore", "fc_wind_offshore"),
)

# %% [markdown]
# Validate the merged table against the data model, which casts the types and drops unlisted columns.

# %%
df = CombinedDataSchema.validate(merged)
df.head(3)

# %% [markdown]
# The merged table passes the data model: one row per hour with no gaps or repeats, every price inside the auction limits, no negative generation forecast, residual load equal to its formula, and empty values only in the neighbour prices, where the model allows them. The averaging has to come before the join. Joined at the raw resolution, three in four forecast rows before October 2025 would have met no price.

# %% [markdown]
# ## 3. Split
#
# Chronological, by German delivery date: train 2019 to 2023, validation 2024, test from
# 2025-01-01. **Everything below explores train only.** Validation is for choosing hyperparameters
# and test is scored once at the end.

# %% [markdown]
# Split by year into train, validation and test, and show the size and dates of each.

# %%
train = df.filter(pl.col("year") <= 2023)
validation = df.filter(pl.col("year") == 2024)
test = df.filter(pl.col("year") >= 2025)

pl.concat(
    [
        part.select(
            set=pl.lit(name),
            hours=pl.len(),
            first_day=pl.col("date").min(),
            last_day=pl.col("date").max(),
        )
        for name, part in [("train", train), ("validation", validation), ("test", test)]
    ]
)

# %% [markdown]
# Train spans three different markets, from the calm years of 2019 and 2020 through the gas crisis of 2021 and 2022 to the recovery in 2023. Validation and test both come after the crisis, so whatever the model learns from crisis prices has to carry over to calmer ones. Test runs to 2026-10-11, 15,575 hours as of this pull, and it is the only set that contains the switch to quarter-hour auctions.

# %% [markdown]
# ## 4. Exploration
#
# ### 4.1 Dataset structure
#
# Gaps, duplicates and value limits are already enforced by the data model. What is left here
# describes the structure rather than tests it.

# %% [markdown]
# Summarise each raw file: rows, date range, duplicate timestamps and where quarter-hour data starts.

# %%
# one row per raw file. quarter_hourly_from is the first timestamp that comes
# 15 minutes after the one before it, which is where the file changes resolution
pl.concat(
    [
        raw.select(
            file=pl.lit(name),
            rows=pl.len(),
            first=pl.col("utc_timestamp").min(),
            last=pl.col("utc_timestamp").max(),
            duplicate_timestamps=pl.col("utc_timestamp").is_duplicated().sum(),
            quarter_hourly_from=pl.col("utc_timestamp")
            .filter(pl.col("utc_timestamp").diff() == pl.duration(minutes=15))
            .min(),
        )
        for name, raw in [
            ("prices", prices_raw),
            ("forecasts", forecasts_raw),
            ("neighbors", neighbors_raw),
        ]
    ]
)

# %% [markdown]
# Count the clock-change days, the forecast hours the join dropped, and the hits on the price floor.

# %%
hours_per_day = df.group_by("date").len()

pl.DataFrame(
    {
        "check": [
            "German days with 23 hours (spring clock change)",
            "German days with 25 hours (autumn clock change)",
            "forecast hours past the last cleared price (dropped)",
            "train hours at the -500 EUR/MWh auction floor",
        ],
        "count": [
            hours_per_day.filter(pl.col("len") == 23).height,
            hours_per_day.filter(pl.col("len") == 25).height,
            # forecast hours later than the last price, which the left join dropped
            forecasts.filter(
                pl.col("utc_timestamp") > df["utc_timestamp"].max()
            ).height,
            train.filter(pl.col("price_da") <= -500).height,
        ],
    }
)

# %% [markdown]
# List the columns that have missing hours.

# %%
df.null_count().unpivot(variable_name="column", value_name="null_hours").filter(
    pl.col("null_hours") > 0
)

# %% [markdown]
# No raw file has a duplicate timestamp. The clock changes land where they should, as eight 23-hour and seven 25-hour German days, which means the 24-hour forecast vector is 23 or 25 long on those days, as `project.md` warns.
#
# Quarter-hour prices start at 00:15 German time on 2025-10-01, inside the test period, so train and validation never see them.
#
# The -500 EUR/MWh floor was hit in one hour in the five train years and needs no special handling. `project.md` expects forecast hours past the last cleared price, and the 2026-10-11 pull has none, because both series end on the same delivery day. The check stays in for the next pull.
#
# Only one gap remains: the Dutch price for all 24 hours of 2026-10-10, the day before the pull and inside test. Most likely it is not yet published, and a refresh before modelling should fill it. The gap only matters if neighbour prices survive as features.

# %% [markdown]
# **What is knowable at 11:55 on D-1** decides which columns can ever be features. From
# `project.md`, checked against the columns that actually exist:
#
# | Column | Day D value usable? | Why |
# |---|---|---|
# | `price_da` | no, it is the target | D-1 and earlier are settled, so lags of 24h and more are legal |
# | `fc_load`, `fc_solar`, `fc_wind_*`, `fc_residual` | yes, with a caveat | Forecasts for D. Computed 08:00 D-1, published 18:00 D-1, so a vendor forecast stands in for them in practice |
# | `price_FR` ... `price_NO2` | **no** | Coupled zones clear in the same auction as DE-LU. Lag 24h or more only |
# | calendar | yes | Known in advance |
#
# No actuals are in the data at all, which removes the most common leak in this problem.

# %% [markdown]
# ### 4.2 The yearly mean price moves eightfold

# %% [markdown]
# Plot the daily mean price over the train period.

# %%
daily = train.group_by("date").agg(price=pl.col("price_da").mean()).sort("date")

(
    ggplot(daily, aes("date", "price"))
    + geom_line(size=0.3)
    + labs(x="", y="daily mean price, EUR/MWh", title="Train period, daily mean price")
)

# %% [markdown]
# Per year: mean price, spread, and the error of the two naive forecasts.

# %%
# absolute error of two naive forecasts: the price 24 and 168 rows earlier.
# the grid is hourly with no gaps, so differencing by rows is differencing by hours
naive_24 = pl.col("price_da").diff(24).abs()
naive_168 = pl.col("price_da").diff(168).abs()

(
    train.with_columns(naive_24=naive_24, naive_168=naive_168)
    .group_by("year")
    .agg(
        mean_price=pl.col("price_da").mean(),
        sd=pl.col("price_da").std(),
        naive_mae_lag24=pl.col("naive_24").mean(),
        naive_mae_lag168=pl.col("naive_168").mean(),
    )
    .with_columns(lag24_error_per_euro=pl.col("naive_mae_lag24") / pl.col("mean_price"))
    .sort("year")
)

# %% [markdown]
# From 30 EUR/MWh in 2020 the yearly mean rose to 235 in 2022, with single days near 700, and fell back to 95 in 2023. The naive error moves with it, from 9 EUR/MWh in 2019 to 58 in 2022, yet as a share of the mean price it stays between 0.25 and 0.33 every year.
#
# Raw MAE therefore cannot compare a CV fold that contains 2022 with a calmer test year, which is why MASE is in the objective. Nothing in the data measures the fuel cost behind the level either. The model has to read the level off recent prices: same-hour lags, yesterday's mean and a longer moving average.

# %% [markdown]
# ### 4.3 The decision only depends on the shape of the day
#
# The company buys a fixed volume and only chooses **which hours**. Shifting all 24 prices of a day
# up or down changes nothing it does. So the variance that matters is the variance *within* a day,
# around that day's mean.

# %% [markdown]
# Split each year's price variance into the part between days and the part within a day.

# %%
# each hour's distance from its own day's mean price
around_daily_mean = train.with_columns(
    daily_mean=pl.col("price_da").mean().over("date"),
).with_columns(deviation=pl.col("price_da") - pl.col("daily_mean"))

# between_days: how much the daily mean moves. within_day: how far hours sit from it.
# within_day_share is the part of the variance the buying decision actually sees
(
    around_daily_mean.group_by("year")
    .agg(
        between_days=pl.col("daily_mean").var(),
        within_day=pl.col("deviation").var(),
    )
    .with_columns(
        within_day_share=pl.col("within_day")
        / (pl.col("within_day") + pl.col("between_days"))
    )
    .sort("year")
)

# %% [markdown]
# Plot the average shape of the day for each year.

# %%
profile = (
    around_daily_mean.group_by("year", "hour")
    .agg(deviation=pl.col("deviation").mean())
    .with_columns(pl.col("year").cast(pl.String))
)

(
    ggplot(profile, aes("hour", "deviation", color="year"))
    + geom_line(size=1)
    + geom_hline(yintercept=0, linetype="dashed")
    + scale_x_continuous(breaks=range(0, 24, 3))
    + labs(
        x="hour of day, German time",
        y="EUR/MWh above the day's mean",
        title="Average shape of the day",
    )
)

# %% [markdown]
# In 2021 and 2022 only 17 to 19% of the price variance sits inside the day. The rest is the level moving from day to day, and the level does not change which hours the company buys. A model trained to minimise error over all hours will spend most of its effort on the part the decision ignores. The evaluation needs a measure of the shape on its own, either the euro cost of the schedule or the error left after removing each day's mean from both the forecast and the actual price.
#
# Inside train, the shape itself changed. In 2019 the day was nearly flat, never more than 12 EUR/MWh from its mean. By 2023 the midday trough at 13:00 to 14:00 sat 26 EUR/MWh below the day's mean, deeper than the night, which is solar pushing midday prices down. The cheapest hours moved from 03:00 to the middle of the day, which leaves a fixed hour-of-day effect learned on 2019 pointing at the wrong hours by 2023. Solar and residual load forecasts are what carry the shift.

# %% [markdown]
# ### 4.4 Residual load sets the shape, and its slope moves with the price level

# %% [markdown]
# Plot the median price against each forecast, one line per year.

# %%
FORECASTS = [
    "fc_load",
    "fc_solar",
    "fc_wind_onshore",
    "fc_wind_offshore",
    "fc_residual",
]

merit = (
    # long format: one row per hour and forecast, so all five share one faceted plot
    train.unpivot(
        index=["year", "price_da"],
        on=FORECASTS,
        variable_name="forecast",
        value_name="mw",
    )
    # cut each forecast into 20 quantile bins with equal hour counts. solar is zero
    # every night, so its lowest quantiles coincide and merge into one bin
    .with_columns(
        bin=pl.col("mw").qcut(20, allow_duplicates=True).over("forecast"),
        year=pl.col("year").cast(pl.String),
    )
    .group_by("forecast", "year", "bin")
    .agg(
        gw=pl.col("mw").median() / 1000,
        price=pl.col("price_da").median(),
        hours=pl.len(),
    )
    .filter(pl.col("hours") >= 50)  # thin bins are noise
)

(
    ggplot(merit, aes("gw", "price", color="year"))
    + geom_line(size=0.8)
    + facet_wrap("~forecast", scales="free_x", ncol=3)
    + labs(x="forecast, GW", y="median price, EUR/MWh")
    + theme(figure_size=(11, 6))
)

# %% [markdown]
# Per year: how residual load relates to the price, overall and within the day, and its slope.

# %%
# price and residual load as distances from their own day's mean
price_dev = pl.col("price_da") - pl.col("price_da").mean().over("date")
residual_dev = pl.col("fc_residual") - pl.col("fc_residual").mean().over("date")

# eur_per_gw is the least-squares slope of price on residual load, scaled from MW to GW
residual_stats = {
    "corr_all_hours": pl.corr("price_da", "fc_residual"),
    "corr_within_day": pl.corr(price_dev, residual_dev),
    "eur_per_gw": pl.cov("price_da", "fc_residual")
    / pl.col("fc_residual").var()
    * 1000,
}

# the same three numbers per year, then once for all train years pooled
pl.concat(
    [
        train.group_by("year")
        .agg(**residual_stats)
        .sort("year")
        .with_columns(pl.col("year").cast(pl.String)),
        train.select(year=pl.lit("2019-2023"), **residual_stats),
    ]
)

# %% [markdown]
# Within-day correlation of each forecast with the price.

# %%
# the within-day correlation from the cell above, once per forecast
train.select(
    [
        pl.corr(price_dev, pl.col(c) - pl.col(c).mean().over("date")).alias(c)
        for c in FORECASTS
    ]
)

# %% [markdown]
# Residual load is the only forecast with a clean rising line in every year. Within a day it correlates 0.76 to 0.85 with the price in each train year, against 0.15 to 0.29 for any single component, which makes it the one fundamental that matters for the shape.
#
# Its slope is the problem. One more GW of residual load added 1.1 EUR/MWh to the price in 2019 and 7.1 in 2022, and 3.1 in 2023. The slope rises and falls with the price level, which points at fuel cost, the one driver missing from the data. Pooled over all five years the correlation drops to 0.39 for the same reason. A linear model with a single coefficient on residual load gets the slope wrong in most years and needs residual load times a recent price level as an interaction. Trees find that interaction on their own. The 2021 line also bends sharply upward above about 52 GW, a scarcity cliff a straight line cannot follow.

# %% [markdown]
# ### 4.5 Price memory

# %% [markdown]
# Correlate the price with itself at every lag up to two weeks.

# %%
# correlation of the price with itself shifted k hours back, for every k up to 14 days
lags = range(1, 24 * 14 + 1)
autocorrelation = pl.DataFrame(
    {
        "lag_hours": list(lags),
        "corr": [
            train.select(pl.corr("price_da", pl.col("price_da").shift(k))).item()
            for k in lags
        ],
    }
)

(
    ggplot(autocorrelation, aes("lag_hours", "corr"))
    + geom_line()
    + geom_point(
        data=autocorrelation.filter(pl.col("lag_hours").is_in([24, 48, 168, 336])),
        color="red",
        size=2.5,
    )
    + scale_x_continuous(breaks=range(0, 24 * 14 + 1, 24))
    + labs(
        x="lag, hours", y="correlation with the price", title="Autocorrelation, train"
    )
)

# %% [markdown]
# Correlation peaks every 24 hours, with extra bumps at 168 and 336 from the weekly cycle. Same hour yesterday reaches 0.90, two days back 0.84 and the same hour last week 0.83. Lag 1 is higher at 0.99, but `project.md` keeps lags same-hour, and it stays out.
#
# Two days back and last week carry about the same correlation, and lag 168 is the one worth keeping, because it lines a Monday up with the previous Monday instead of with Sunday. The naive lag-24 forecast also beat lag 168 in every train year (table in 4.2), which makes lag 24 the baseline to beat.

# %% [markdown]
# ### 4.6 Neighbouring zones
#
# Three numbers per zone. **Same hour** is a leak and is shown only to measure how tightly the zones
# are coupled. **Lag 24** is the legal version. **Spread** asks whether yesterday's gap between the
# zone and Germany predicts how much the German price changes today.

# %% [markdown]
# Per zone: how its price relates to Germany's, at the same hour, a day back, and as a spread.

# %%
NEIGHBORS = [c for c in df.columns if c.startswith("price_") and c != "price_da"]

# every price shifted back 24 hours, as new *_lag24 columns
lagged = train.with_columns(
    pl.col(["price_da", *NEIGHBORS]).shift(24).name.suffix("_lag24")
)
# how much the German price moved since the same hour yesterday
change = pl.col("price_da").diff(24)

# same_hour_LEAK: the zone's price in the same hour, not knowable at the gate
# lag24: the zone's price a day earlier
# spread_vs_change: does yesterday's gap (zone minus Germany) predict today's change?
# Germany's own row only has lag24, as the number to beat
pl.DataFrame(
    {
        "zone": ["DE-LU itself", *NEIGHBORS],
        "same_hour_LEAK": [
            None,
            *[lagged.select(pl.corr("price_da", z)).item() for z in NEIGHBORS],
        ],
        "lag24": [
            lagged.select(pl.corr("price_da", f"{z}_lag24")).item()
            for z in ["price_da", *NEIGHBORS]
        ],
        "spread_vs_change": [
            None,
            *[
                lagged.select(
                    pl.corr(change, pl.col(f"{z}_lag24") - pl.col("price_da_lag24"))
                ).item()
                for z in NEIGHBORS
            ],
        ],
    }
)

# %% [markdown]
# Plot the same-hour price correlation between every pair of zones.

# %%
ZONES = ["price_da", *NEIGHBORS]
LABELS = ["DE-LU" if z == "price_da" else z.removeprefix("price_") for z in ZONES]
coupling = (
    train.select([pl.col(z).alias(label) for z, label in zip(ZONES, LABELS)])
    .drop_nulls()
    .corr()
    .with_columns(zone=pl.Series(LABELS))
    .unpivot(index="zone", variable_name="other", value_name="corr")
)

(
    ggplot(coupling, aes("zone", "other", fill="corr"))
    + geom_tile()
    + geom_text(aes(label="corr"), format_string="{:.2f}", size=7)
    + scale_fill_gradient(low="white", high="steelblue")
    + labs(x="", y="", title="Same-hour price correlation between zones, train")
    + theme(axis_text_x=element_text(rotation=45, ha="right"), figure_size=(8, 7))
)

# %% [markdown]
# Same-hour coupling is tight: 0.97 to 0.98 for the Czech Republic, Austria, the Netherlands, Belgium and DK1. That is why a same-hour neighbour price would be such a damaging leak. Lagged a day, no neighbour beats Germany's own lag 24 (0.903). The best are the Czech Republic at 0.906 and Belgium at 0.904, so as plain lags the neighbours add almost nothing.
#
# Spreads are the open question. When a neighbour was dearer than Germany yesterday, the German price tends to rise today, at 0.31 for Switzerland and 0.28 to 0.30 for the Netherlands and southern Norway (NO2). That may only be Germany recovering after an unusually cheap day, which its own price history already shows. A 15-minute check settles it, and it is listed at the checkpoint.
#
# For linear models, the central-western zones correlate 0.94 to 0.98 with each other and behave like one variable repeated six times.

# %% [markdown]
# ### 4.7 Negative prices

# %% [markdown]
# Count negative-price hours by hour of day.

# %%
negatives = (
    train.group_by("hour")
    .agg(negative_hours=(pl.col("price_da") < 0).sum())
    .sort("hour")
)

(
    ggplot(negatives, aes("hour", "negative_hours"))
    + geom_col()
    + scale_x_continuous(breaks=range(0, 24, 3))
    + labs(
        x="hour of day, German time",
        y="hours below 0 EUR/MWh",
        title="Negative prices, train",
    )
)

# %% [markdown]
# Per year: negative hours, lowest and highest price.

# %%
train.group_by("year").agg(
    negative_hours=(pl.col("price_da") < 0).sum(),
    lowest=pl.col("price_da").min(),
    highest=pl.col("price_da").max(),
).sort("year")

# %% [markdown]
# 1,018 train hours cleared below zero, 2.3% of the total, in two clusters: midday from 12:00 to 15:00, which is solar, and the night from 01:00 to 04:00, which is wind meeting low demand. The crisis pushed the count down to 69 in 2022, and it was back to 301 in 2023.
#
# These are the hours where getting the ranking right pays most, because the company is paid to take the power. A log transform is off the table, since it cannot take a negative price. asinh is the option if diagnostics later show the error growing with the level. The single -500 hour stays in as a genuine auction outcome.

# %% [markdown]
# ### 4.8 Calendar

# %% [markdown]
# Compare each weekday's mean price with the mean of its week.

# %%
(
    train.group_by("date", "weekday")
    .agg(daily_mean=pl.col("price_da").mean())
    # each day's distance from the mean of its calendar week
    .with_columns(week=pl.col("date").dt.truncate("1w"))
    .with_columns(
        vs_week=pl.col("daily_mean") - pl.col("daily_mean").mean().over("week")
    )
    # weekday 1 is Monday. share_of_weeks_below_mean: how often that day is the cheaper side
    .group_by("weekday")
    .agg(
        eur_vs_week_mean=pl.col("vs_week").mean(),
        share_of_weeks_below_mean=(pl.col("vs_week") < 0).mean(),
    )
    .sort("weekday")
)

# %% [markdown]
# Sunday averages 27 EUR/MWh below its week's mean and sits below it in 89% of weeks. Saturday is 15 below, and Tuesday to Thursday 9 to 12 above. Monday is only 5 above, likely because its first hours still follow the quiet Sunday night.
#
# Weekday dummies will pick this up. Public holidays will not: a holiday on a Wednesday looks like a Sunday to the market and like a Wednesday to the calendar. Holidays are not in the data, which makes them a side exploration.

# %% [markdown]
# ### 4.9 Collinearity among the forecasts

# %% [markdown]
# Correlation matrix of the five forecasts.

# %%
train.select(FORECASTS).corr().with_columns(forecast=pl.Series(FORECASTS)).select(
    "forecast", *FORECASTS
)

# %% [markdown]
# Pairwise, only onshore and offshore wind move together (0.71). The bigger issue does not show in a correlation table. `fc_residual` is load minus the other three exactly, which makes the five columns perfectly collinear: OLS cannot estimate all five at once, and Lasso picks between them arbitrarily. Linear models get residual load plus at most solar, which carries the time-of-day shift. Trees can take all five.

# %% [markdown]
# ## 5. Checkpoint
#
# Phase 1 stops here. Cleaning, features and models wait for a go-ahead and any edits to the lists below.
#
# ### Discoveries, ranked by how much they should change the model
#
# 1. The decision only uses the shape of the day, and in 2021 and 2022 over 80% of the price variance is the level. Evaluate the shape on its own, through the euro cost of the schedule and the error left after removing each day's mean, next to MAE and RMSE.
# 2. Residual load drives the shape, at a within-day correlation of 0.76 to 0.85 in every train year, but its slope went from 1.1 to 7.1 EUR/MWh per GW between 2019 and 2022. Linear models need residual load times a recent price level.
# 3. Yearly mean prices moved eightfold, and nothing in the data measures fuel cost. The level has to come from price memory: lags of 24, 48 and 168 hours, yesterday's mean and a 30-day mean. MASE goes next to MAE so years at different levels compare.
# 4. The shape of the day changed inside train. As solar grew, the cheapest hours moved from 03:00 to 13:00, and hour-of-day effects learned on the early years go stale. Solar forecasts or per-hour models have to carry the shift.
# 5. Same hour yesterday is the strongest single predictor at 0.90 and beat same hour last week in every train year. It is the baseline.
# 6. Weekends shift the level: Sunday sits 27 EUR/MWh and Saturday 15 below the week's mean. Weekday dummies cover it. Holidays are missing.
# 7. Residual load is an exact combination of the other four forecasts. Linear models take residual load and solar, not all five.
# 8. Neighbour prices add nothing as plain lags, with the best at 0.906 against Germany's own 0.903. Only the spread might, and that is unproven.
# 9. 2.3% of train hours are negative, clustered at midday and at night. No log transform, and asinh if one is needed.
#
# ### Proposed cleaning
#
# 1. Drop the first 30 days of train, which have no 30-day mean and no lag 168 yet.
# 2. Refresh the neighbour file before modelling to fill the missing Dutch day. If it is still empty and neighbours stay in, forward-fill from the last published hour, which only looks backwards.
# 3. Keep every extreme. The -500 hour and the 2022 spikes are auction results, not errors.
# 4. No target transform for now. asinh is the candidate if the diagnostics show error growing with the price level.
# 5. Keep the hourly mean of the four quarter-hour prices as the target after 2025-10-01, so train and test predict the same thing.
#
# ### Worth exploring, not core
#
# 1. Model the level and the shape separately, about an hour. Predict the daily mean, then the 24 deviations from it. The decision only needs the second part, which is also the more stable one (discoveries 1 and 4).
# 2. Test the neighbour spread against Germany's own recovery, 15 minutes. If Switzerland's 0.31 disappears once Germany's distance from its 7-day mean is in the model, the 11 neighbour columns and their 25-minute pull can go.
# 3. Add a gas price, half a day to find a free daily TTF series. It would replace the 30-day-mean stand-in for the level and explain the moving slope, which likely makes it the biggest single gain in MAE, but it does nothing for the shape.
# 4. Flag German public holidays, 20 minutes with the `holidays` package. About ten weekdays a year behave like Sundays and the weekday dummies get them wrong.
#
# ### Model shortlist
#
# | Model | Why it fits these findings | Needs from features | What a win or a loss tells us |
# |---|---|---|---|
# | Same hour yesterday | Strongest single predictor at 0.90 | Nothing | The bar. A model that loses to it has no story |
# | Training-set mean | Required by `project.md` | Nothing | The floor |
# | Lasso | Reads the level off lags, so it can predict above the range it trained on. Picks among correlated lags. The standard benchmark in price forecasting (LEAR) | Scaling, residual load times level, hour and weekday dummies, no exactly collinear set | A win says the shape is mostly linear once the slope can move |
# | Gradient boosting (LightGBM or XGBoost) | Fits the moving slope and the 52 GW cliff without hand-built interactions | Lags for the level, since trees cannot predict outside the range they saw | A win says the interactions matter more than we can build by hand. A loss says the level problem dominates |
# | Lasso, then boosting on its errors | The level from the linear part, the bends from the trees | Both of the above | A win means the two failure modes are separate and can be split |
# | Random forest | Covers the course's tree family | As boosting | Mostly a comparison point for boosting |
# | Small neural net (MLP) | Covers the course's neural-net family | Scaling | Least likely to win on 44,000 rows that span a regime change |
#
# `project.md` also suggests 24 separate models, one per delivery hour. That is a variant of whichever model wins rather than a model of its own, and it answers discovery 4 directly.
