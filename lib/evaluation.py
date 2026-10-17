"""Scoring: the price error, the cost of the shift a forecast picks, and the fold loop.

Every function works on the hourly table built in notebook section 4.2, which carries
`date`, `hour`, `weekday`, `price_da` and `price_lag24`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import numpy as np
import polars as pl

from lib.decision import SHIFT_HOURS, SHIFT_MW, STANDARD_SHIFT, best_shift, premium
from lib.modeling import Regressor, df_to_np
from lib.splitting import make_folds

# one period's scores by name; None where a metric does not apply, such as MAE for a rule
Scores = dict[str, float | None]


def evaluate(
    frame: pl.DataFrame,
    score: np.ndarray,
    pick_with_premium: bool = True,
    report_error: bool = True,
) -> Scores:
    """Score one period: price error, and the cost of the shift the score picks.

    pick_with_premium: add the premium to the score before picking the shift. False for
    a score that is not in EUR/MWh, such as residual load.
    report_error: compute MAE, RMSE and MASE. False for a score that is not an hourly
    price forecast, such as last year's average profile or residual load.
    """
    cost = pl.col("price_da") + premium()
    pick_on = pl.col("score") + premium() if pick_with_premium else pl.col("score")
    first, last = STANDARD_SHIFT
    error = pl.col("price_da") - pl.col("score")
    out = (
        frame.with_columns(score=pl.Series(score))
        .with_columns(
            picked=best_shift(pick_on),
            hindsight=best_shift(cost),
            standard=pl.col("hour").is_between(first, last),
        )
        .select(
            mae=error.abs().mean(),
            rmse=(error**2).mean().sqrt(),
            naive_mae=(pl.col("price_da") - pl.col("price_lag24")).abs().mean(),
            shift_cost=cost.filter(pl.col("picked")).mean(),
            standard_cost=cost.filter(pl.col("standard")).mean(),
            best_cost=cost.filter(pl.col("hindsight")).mean(),
            # a day counts as right when the picked shift is the cheapest one
            right_shift_share=(pl.col("picked") == pl.col("hindsight"))
            .all()
            .over("date")
            .mean(),
        )
        .row(0, named=True)
    )
    return {
        "mae": out["mae"] if report_error else None,
        "rmse": out["rmse"] if report_error else None,
        "mase": out["mae"] / out["naive_mae"] if report_error else None,
        "shift_cost": out["shift_cost"],
        "saving_share": (out["standard_cost"] - out["shift_cost"])
        / (out["standard_cost"] - out["best_cost"]),
        "right_shift_share": out["right_shift_share"],
        # what the picked shift saves against the standard shift for a 500 MWh day, over a year
        "saving_eur_per_year": (out["standard_cost"] - out["shift_cost"])
        * SHIFT_MW
        * SHIFT_HOURS
        * 365,
    }


def cross_validate(
    model: Regressor, columns: list[str], data: pl.DataFrame, folds: dict[str, int]
) -> tuple[Scores, pl.DataFrame]:
    """Fit a fresh copy of the model on each fold's training years, score its validation
    year, and average over the folds. The model passed in stays unfitted.

    Returns the averaged scores, with `saving_share_fold_sd` and `mae_fold_sd` as the
    noise between folds, and the out-of-fold predictions for diagnostics.
    """
    scores: list[Scores] = []
    oof: list[pl.DataFrame] = []
    for fit, valid in make_folds(data, folds).values():
        fold_model = deepcopy(model)
        fold_model.fit(df_to_np(fit, columns), fit["price_da"].to_numpy())
        pred = fold_model.predict(df_to_np(valid, columns))
        scores.append(evaluate(valid, pred))
        oof.append(valid.with_columns(pred=pl.Series(pred)))
    return summarise_folds(scores), pl.concat(oof)


def summarise_folds(scores: list[Scores]) -> Scores:
    """Average each fold's scores, adding `saving_share_fold_sd` and `mae_fold_sd` as the
    noise between folds."""
    table = pl.DataFrame(scores)
    # Series.mean and .std are typed as any scalar; on these float columns they are floats
    summary: Scores = {k: cast(float | None, table[k].mean()) for k in table.columns}
    summary["saving_share_fold_sd"] = cast(float | None, table["saving_share"].std())
    summary["mae_fold_sd"] = cast(float | None, table["mae"].std())
    return summary
