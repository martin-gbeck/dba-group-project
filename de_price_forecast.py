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
#     display_name: Python 3.12.0 (spot-price-forecast; uv)
#     language: python
#     name: Python 3.12.0 (spot-price-forecast; uv)
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
# Mechanical fixes only, the same for every file:
#
# 1. Every file is averaged to UTC hours. Prices go quarter-hourly on 2025-10-01 and the forecasts are quarter-hourly throughout, so the mean puts all three on one grid. It also fills Switzerland, which stays hourly and leaves three of every four quarter-hours empty.
# 2. The join starts from the price table, so every row has a target.
# 3. Calendar columns are in German local time, because demand follows the local clock.
# 4. `fc_residual` is load minus solar and wind, as defined in `project.md`.
# 5. The result is validated against the data model in `data_schemas.py`. Read that file for what the table holds.

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
# The averaging has to come before the join: at raw resolution, three in four forecast rows before October 2025 would have no price to join to.

# %% [markdown]
# ## 3. Split
#
# Chronological by delivery date, with two validation folds so hyperparameters are chosen on more than one year:
#
# | Fold | Train | Validate |
# |---|---|---|
# | 1 | 2019 to 2022 | 2023 |
# | 2 | 2019 to 2023 | 2024 |
#
# Fold scores are averaged in MASE, since 2023 and 2024 sit at different price levels. The chosen model is then refit on 2019 to 2024 and scores the test set, 2025 onward, once.
#
# Exploration below uses 2019 to 2023, everything fold 2 trains on. That makes fold 1's score slightly optimistic: the features are shaped partly by looking at 2023, its validation year. Fold 2's 2024 was never explored.

# %% [markdown]
# Define the two folds and the test set, and show the size and dates of each.

# %%
LAST_EXPLORED_YEAR = 2023
TEST_FROM = 2025
FOLDS = {
    "fold 1": 2023,
    "fold 2": 2024,
}  # fold name: validation year, trained on all years before it

# exploration works on everything fold 2 trains on
train = df.filter(pl.col("year") <= LAST_EXPLORED_YEAR)
test = df.filter(pl.col("year") >= TEST_FROM)
folds = {
    name: (df.filter(pl.col("year") < year), df.filter(pl.col("year") == year))
    for name, year in FOLDS.items()
}

# one row per piece: each fold's training and validation data, then the test set
pieces = [
    (f"{name} {role}", part)
    for name, (fit, valid) in folds.items()
    for role, part in (("train", fit), ("validate", valid))
] + [("test", test)]
pl.DataFrame(
    [
        {
            "set": label,
            "hours": part.height,
            "first_day": part["date"].min(),
            "last_day": part["date"].max(),
        }
        for label, part in pieces
    ]
)

# %% [markdown]
# Train covers three markets: calm (2019 to 2020), the gas crisis (2021 to 2022) and the recovery (2023). Fold 1 validates on the first year after the crisis and fold 2 on a calm one, so a setting has to work in both. Only test contains quarter-hour auctions.

# %% [markdown]
# ## 4. Exploration
#
# ### 4.1 Dataset structure
#
# The data model already tests gaps, duplicates and limits. These cells only describe the structure.

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
# Nothing here needs fixing. On the 23- and 25-hour clock-change days the forecast vector is 23 or 25 long, not 24. The -500 floor was hit once in train and needs no special handling. Only the Dutch price for 2026-10-10 is missing, most likely not yet published, so refresh before modelling.

# %% [markdown]
# Which columns can be features at 11:55 on D-1, from `project.md`:
#
# | Column | Day D value usable? | Why |
# |---|---|---|
# | `price_da` | no, it is the target | D-1 and earlier are settled, so lags of 24h and more are legal |
# | `fc_load`, `fc_solar`, `fc_wind_*`, `fc_residual` | yes, with a caveat | Computed 08:00 D-1 but published 18:00 D-1, so in practice a vendor forecast stands in |
# | `price_FR` ... `price_NO2` | **no** | They clear in the same auction as DE-LU. Lag 24h or more only |
# | calendar | yes | Known in advance |
#
# The data holds no actuals, which removes the most common leak in this problem.

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
# The naive error grows with the price level, but as a share of the mean it holds at 0.25 to 0.33 every year. MAE alone cannot compare a CV fold that contains 2022 with a calm test year, which is why MASE goes beside it. No column measures fuel cost, so the level has to come from recent prices: lags, yesterday's mean and a longer moving average.

# %% [markdown]
# ### 4.3 The decision only depends on the shape of the day
#
# The company only chooses which hours to buy, so moving a whole day's prices up or down changes nothing it does. What matters is the variance within a day.

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
# In the crisis years only 17 to 19% of the variance sits within the day, the part the decision uses. A model that minimises error over all hours spends most of its effort on the level, so the evaluation needs a shape measure: the euro cost of the schedule, or the error left after removing each day's mean.
#
# The shape moved as well. By 2023 solar had pushed the cheapest hours from 03:00 to 13:00 to 14:00, so an hour-of-day effect learned on early years points at the wrong hours. The solar and residual-load forecasts have to carry that shift.

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
# Residual load is the forecast that drives the shape: 0.76 to 0.85 within-day correlation in every year, against at most 0.29 for any single component. Its slope moves with the price level, 1.1 EUR/MWh per GW in 2019 against 7.1 in 2022, which is why the pooled correlation falls to 0.39. A linear model needs residual load times a recent price level; trees find that on their own. The 2021 line also bends upward above about 52 GW, a scarcity cliff no straight line follows.

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
# Same hour yesterday (0.90) is the strongest lag, and it beat same hour last week as a forecast in every train year, so it is the baseline. Lag 1 is higher but stays out under `project.md`'s same-hour rule. Of the rest, lag 168 is worth keeping over lag 48 because it lines Monday up with Monday rather than with Sunday.

# %% [markdown]
# ### 4.6 Neighbouring zones
#
# Same hour is a leak, shown only to measure coupling. Lag 24 is the legal version. Spread asks whether yesterday's gap between a zone and Germany predicts today's German change.

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
# Same-hour coupling reaches 0.98, which is what would make a same-hour neighbour price such a damaging leak. Lagged a day, no zone beats Germany's own lag 24, so plain neighbour lags add nothing.
#
# The spread is the open question: it predicts today's German change at up to 0.31 (Switzerland), but that may just be Germany reverting after a cheap day, which its own history already captures. A 15-minute check at the checkpoint settles it. For a linear model the central-western zones, at 0.94 to 0.98 with each other, are one variable repeated six times.

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
# Negative hours cluster at midday, which is solar, and at night, which is wind meeting low demand. They are where ranking pays most, because the company is paid to take the power. A log transform cannot handle them; asinh can, if diagnostics later ask for a transform.

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
# Weekends lower the level: Sunday sits 27 EUR/MWh below its week and below it in 89% of weeks. Weekday dummies cover that. Public holidays are not in the data and would look like ordinary weekdays to the model, which makes them a side exploration.

# %% [markdown]
# ### 4.9 Collinearity among the forecasts

# %% [markdown]
# Correlation matrix of the five forecasts.

# %%
train.select(FORECASTS).corr().with_columns(forecast=pl.Series(FORECASTS)).select(
    "forecast", *FORECASTS
)

# %% [markdown]
# Only the two wind forecasts correlate (0.71), but `fc_residual` is an exact combination of the other four, so OLS cannot fit all five. Linear models take residual load and solar; trees can take all five.

# %% [markdown]
# ## 5. Checkpoint
#
# Phase 1 stops here. Cleaning, features and models wait for a go-ahead.
#
# ### Discoveries, ranked by how much they should change the model
#
# 1. The decision only uses the shape of the day, and in 2021 and 2022 over 80% of the variance is the level. Score the shape on its own (euro cost of the schedule, error after removing the daily mean) next to MAE and RMSE.
# 2. Residual load drives the shape, but its slope went from 1.1 to 7.1 EUR/MWh per GW. Linear models need residual load times a recent price level.
# 3. The yearly mean moved eightfold and no column measures fuel cost. Price lags (24, 48, 168 hours), yesterday's mean and a 30-day mean have to carry it, with MASE beside MAE.
# 4. Solar moved the cheapest hours from 03:00 to 13:00 inside train, so fixed hour effects go stale. The solar forecast or per-hour models carry the shift.
# 5. Same hour yesterday is the baseline: strongest lag at 0.90, and it beat same hour last week every year.
# 6. Weekends lower the level by 15 to 27 EUR/MWh. Weekday dummies cover it; holidays are missing.
# 7. Residual load is an exact combination of the other four forecasts. Linear models take residual load and solar.
# 8. Neighbour prices add nothing as plain lags. Only the spread might, unproven.
# 9. 2.3% of train hours are negative, so no log transform; asinh if one is needed.
#
# ### Proposed cleaning
#
# 1. Drop the first 30 days of train, which lack a 30-day mean and lag 168.
# 2. Refresh the neighbour file to fill the missing Dutch day; if still empty and neighbours stay in, forward-fill.
# 3. Keep every extreme. The -500 hour and the 2022 spikes are auction results.
# 4. No target transform for now; asinh if diagnostics show error growing with the level.
# 5. Keep the hourly mean of the four quarter-hour prices as the target after 2025-10-01.
#
# ### Worth exploring, not core
#
# 1. Separate level and shape models, about an hour: predict the daily mean, then the 24 deviations. The decision needs only the second, which is also the more stable.
# 2. Neighbour spread against Germany's own reversion, 15 minutes. If it adds nothing, the neighbour columns and their 25-minute pull go.
# 3. A gas price series, half a day to source. Likely the biggest MAE gain, but no help with the shape.
# 4. German public holidays, 20 minutes with the `holidays` package. About ten weekdays a year price like Sundays.
#
# ### Model shortlist
#
# | Model | Why it fits | Needs from features | A win or loss tells us |
# |---|---|---|---|
# | Same hour yesterday | Strongest lag | Nothing | The bar to beat |
# | Training-set mean | Required by `project.md` | Nothing | The floor |
# | Lasso | Reads the level off lags, so it extrapolates; standard price-forecasting benchmark (LEAR) | Scaling, residual load times level, hour and weekday dummies, no collinear set | Win: the shape is mostly linear once the slope can move |
# | Gradient boosting | Fits the moving slope and the 52 GW cliff without hand-built interactions | Lags for the level, since trees cannot extrapolate | Win: interactions matter. Loss: the level problem dominates |
# | Lasso, then boosting on its errors | Level from the linear part, bends from the trees | Both | Win: the two failure modes split cleanly |
# | Random forest | Course tree family | As boosting | A comparison point for boosting |
# | Small MLP | Course neural-net family | Scaling | Least likely to win on 44,000 rows across a regime change |
#
# Per-hour models, one per delivery hour as `project.md` suggests, are a variant of whichever model wins, and they answer discovery 4 directly.
