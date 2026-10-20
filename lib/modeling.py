"""The forecasting method and what it needs: Lasso-then-XGBoost, the asinh wrapper, and
the step that turns table columns into the matrix a model takes."""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

CATEGORICAL = ["hour", "weekday", "month"]  # one-hot in the Lasso, raw in the trees


class Regressor(Protocol):
    """Any class with fit and predict on numpy arrays, such as a scikit-learn or xgboost model."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def df_to_np(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """The model input: the named columns as a float matrix."""
    return frame.select(columns).cast(pl.Float64).to_numpy()


class LassoThenXGB:
    """A Lasso on the linear columns predicts the price; XGBoost on all columns predicts its error.

    The Lasso fits the steady pattern in the price; the trees fit the less regular part it
    misses. The prediction is the sum of the two.
    """

    def __init__(
        self,
        columns: list[str],
        linear_columns: list[str],
        alpha: float = 1.0,
        **xgb_settings: Any,
    ):
        """columns: every column of the matrix the model gets, in order.
        linear_columns: the subset the Lasso uses, with the calendar one-hot encoded and
        the rest standardised. xgb_settings go to XGBRegressor.
        """
        categorical = [
            i
            for i, column in enumerate(columns)
            if column in linear_columns and column in CATEGORICAL
        ]
        numeric = [
            i
            for i, column in enumerate(columns)
            if column in linear_columns and column not in CATEGORICAL
        ]
        linear_prep = ColumnTransformer(
            [
                ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("scale", StandardScaler(), numeric),
            ]
        )
        self.lasso = make_pipeline(linear_prep, Lasso(alpha=alpha, max_iter=20_000))
        self.xgb = XGBRegressor(random_state=0, **xgb_settings)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LassoThenXGB:
        self.lasso.fit(X, y)
        self.xgb.fit(X, y - self.lasso.predict(X))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lasso.predict(X) + self.xgb.predict(X)


class AsinhTarget:
    """Fits the inner model on asinh(price) and maps its predictions back with sinh.

    asinh(x) = ln(x + sqrt(x² + 1)) compresses large prices like a log but, unlike a log,
    handles negative ones.
    """

    def __init__(self, inner: Regressor):
        self.inner = inner

    def fit(self, X: np.ndarray, y: np.ndarray) -> AsinhTarget:
        self.inner.fit(X, np.arcsinh(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sinh(self.inner.predict(X))


LEAR_LAG_DAYS = (1, 2, 3, 7)  # price lags: all 24 hours of D-1, D-2, D-3 and D-7
LEAR_EXOG_DAYS = (0, 1, 7)  # load and renewable forecasts of D, D-1 and D-7
LEAR_CALIBRATION_YEARS = 2


def lear_inputs(hourly: pl.DataFrame) -> pl.DataFrame:
    """One row per delivery day: its 24 prices and the LEAR inputs, all known at 11:55 on D-1.

    Inputs are every hour of the price on D-1, D-2, D-3 and D-7, every hour of the load
    and renewable forecasts on D, D-1 and D-7, and the weekday. A clock-change day's
    missing hour takes the hour before it; its doubled hour is averaged.
    """
    long = (
        hourly.with_columns(
            renewables=pl.col("fc_solar")
            + pl.col("fc_wind_onshore")
            + pl.col("fc_wind_offshore")
        )
        .group_by("date", "hour")
        .agg(
            pl.col("price_da").mean(),
            pl.col("fc_load").mean(),
            pl.col("renewables").mean(),
        )
    )
    grid = (
        long.select("date")
        .unique()
        .join(
            pl.DataFrame({"hour": list(range(24))}, schema={"hour": pl.Int8}),
            how="cross",
        )
    )
    long = (
        grid.join(long, on=["date", "hour"], how="left")
        .sort("date", "hour")
        .with_columns(pl.col("price_da", "fc_load", "renewables").forward_fill())
    )
    wide = long.pivot(
        on="hour", index="date", values=["price_da", "fc_load", "renewables"]
    )
    hours = range(24)
    days = wide.select("date", *[f"price_da_{hour}" for hour in hours])
    for lag in LEAR_LAG_DAYS:
        shifted = wide.select(
            (pl.col("date") + pl.duration(days=lag)).alias("date"),
            *[
                pl.col(f"price_da_{hour}").alias(f"lag_price_d{lag}_{hour}")
                for hour in hours
            ],
        )
        days = days.join(shifted, on="date", how="left")
    for lag in LEAR_EXOG_DAYS:
        shifted = wide.select(
            (pl.col("date") + pl.duration(days=lag)).alias("date"),
            *[pl.col(f"fc_load_{hour}").alias(f"load_d{lag}_{hour}") for hour in hours],
            *[
                pl.col(f"renewables_{hour}").alias(f"renew_d{lag}_{hour}")
                for hour in hours
            ],
        )
        days = days.join(shifted, on="date", how="left")
    weekday = pl.col("date").dt.weekday()
    return days.with_columns(
        [
            (weekday == day).cast(pl.Float64).alias(f"weekday_{day}")
            for day in range(1, 8)
        ]
    ).sort("date")


def lear_forecast(fit: pl.DataFrame, predict: pl.DataFrame) -> np.ndarray:
    """LEAR price forecast for every hour of `predict`, trained on the last two years of `fit`.

    One Lasso per delivery hour, its penalty chosen by AIC. Prices, the target and the
    price lags alike, are scaled by the training median and median absolute deviation
    and passed through asinh; forecasts are standardised. Returns one prediction per
    row of `predict`, in its row order.
    """
    days = lear_inputs(pl.concat([fit, predict], how="diagonal_relaxed"))
    # Series.max is typed as any scalar; the year column is an integer
    first_fit_year = cast(int, fit["year"].max()) - LEAR_CALIBRATION_YEARS + 1
    fit_dates = fit.filter(pl.col("year") >= first_fit_year)["date"].unique()
    train = days.filter(pl.col("date").is_in(fit_dates.implode())).drop_nulls()
    test = days.filter(pl.col("date").is_in(predict["date"].unique().implode()))

    targets = [f"price_da_{hour}" for hour in range(24)]
    price_inputs = [
        column for column in days.columns if column.startswith("lag_price_")
    ]
    exog_inputs = [
        column for column in days.columns if column.startswith(("load_", "renew_"))
    ]
    dummies = [column for column in days.columns if column.startswith("weekday_")]

    prices = train.select(targets).to_numpy().ravel()
    median = float(np.median(prices))
    mad = float(np.median(np.abs(prices - median))) or 1.0

    def stabilise(x: np.ndarray) -> np.ndarray:
        return np.arcsinh((x - median) / mad)

    exog_mean = train.select(exog_inputs).to_numpy().mean(axis=0)
    exog_sd = train.select(exog_inputs).to_numpy().std(axis=0) + 1e-9

    def design_matrix(frame: pl.DataFrame) -> np.ndarray:
        return np.hstack(
            [
                stabilise(frame.select(price_inputs).to_numpy()),
                (frame.select(exog_inputs).to_numpy() - exog_mean) / exog_sd,
                frame.select(dummies).to_numpy(),
            ]
        )

    x_train, x_test = design_matrix(train), design_matrix(test)
    rows: list[pl.DataFrame] = []
    for hour, target in enumerate(targets):
        model = LassoLarsIC(criterion="aic", max_iter=2_000)
        model.fit(x_train, stabilise(train[target].to_numpy()))
        pred = np.sinh(model.predict(x_test)) * mad + median
        rows.append(
            pl.DataFrame(
                {
                    "date": test["date"],
                    "hour": pl.Series([hour] * test.height, dtype=pl.Int8),
                    "lear": pred,
                }
            )
        )
    by_hour = pl.concat(rows)
    joined = (
        predict.select("date", "hour")
        .with_row_index("row")
        .join(by_hour, on=["date", "hour"], how="left")
    )
    return joined.sort("row")["lear"].to_numpy()
