"""The forecasting method and what it needs: Lasso-then-XGBoost, the asinh wrapper, and
the step that turns table columns into the matrix a model takes."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
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
            i for i, c in enumerate(columns) if c in linear_columns and c in CATEGORICAL
        ]
        numeric = [
            i
            for i, c in enumerate(columns)
            if c in linear_columns and c not in CATEGORICAL
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
