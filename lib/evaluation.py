"""Scoring: the price error, the cost of the shift a forecast picks, and the fold loop.

Every function works on the hourly table built in notebook section 4.2, which carries
`date`, `hour`, `weekday`, `price_da` and `price_lag24`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TypedDict, cast

import numpy as np
import polars as pl

from lib.decision import SHIFT_HOURS, SHIFT_MW, STANDARD_SHIFT, best_shift, premium
from lib.modeling import Regressor, df_to_np
from lib.splitting import make_folds

# a score, or None where it does not apply, such as MAE for the residual-load rule
type Metric = float | None


class Scores(TypedDict):
    """One period's scores. MAE, RMSE and MASE are None for a score that is not an hourly
    price forecast, such as residual load."""

    mae: Metric  # EUR/MWh
    rmse: Metric  # EUR/MWh
    mase: Metric  # MAE over that of same hour yesterday
    shift_cost: float  # EUR/MWh, electricity plus premium, of the picked shifts
    # share of the possible saving: (standard - picked) / (standard - hindsight),
    # each a shift's cost in EUR/MWh
    saving_share: float
    right_shift_share: float  # share of days on which the picked shift is the cheapest
    # what the picked shift saves against the standard shift over a year, for 500 MWh a day
    saving_eur_per_year: float


class CVScores(Scores):
    """Scores averaged over the validation years, with how far two of them move from
    year to year."""

    saving_share_fold_sd: Metric  # SD of the yearly saving shares
    mae_fold_sd: Metric  # SD of the yearly MAEs


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
    return Scores(
        mae=out["mae"] if report_error else None,
        rmse=out["rmse"] if report_error else None,
        mase=out["mae"] / out["naive_mae"] if report_error else None,
        shift_cost=out["shift_cost"],
        saving_share=(out["standard_cost"] - out["shift_cost"])
        / (out["standard_cost"] - out["best_cost"]),
        right_shift_share=out["right_shift_share"],
        saving_eur_per_year=(out["standard_cost"] - out["shift_cost"])
        * SHIFT_MW
        * SHIFT_HOURS
        * 365,
    )


def cross_validate(
    model: Regressor, columns: list[str], data: pl.DataFrame, folds: dict[str, int]
) -> tuple[CVScores, pl.DataFrame]:
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


def summarise_folds(scores: list[Scores]) -> CVScores:
    """Average the validation years' scores and add the SD of the saving share and of MAE
    across those years."""
    table = pl.DataFrame(scores)
    summary = {column: table[column].mean() for column in table.columns}
    summary["saving_share_fold_sd"] = table["saving_share"].std()
    summary["mae_fold_sd"] = table["mae"].std()
    # the table's columns are exactly the keys of Scores, so the averages plus the two
    # SDs make a CVScores
    return cast(CVScores, summary)


def block_bootstrap_means(
    values: np.ndarray, block_days: int, samples: int = 10_000, seed: int = 0
) -> np.ndarray:
    """The means of `samples` resampled series, each as long as `values` and built from runs
    of `block_days` consecutive values drawn at random with replacement. A run keeps the
    weekly cycle and the streaks of similar days that drawing single days would break;
    `block_days=1` draws single days."""
    n = len(values)
    rng = np.random.default_rng(seed)
    runs_per_sample = -(-n // block_days)  # rounded up, then trimmed to n
    starts = rng.integers(0, n - block_days + 1, size=(samples, runs_per_sample))
    positions = (starts[:, :, None] + np.arange(block_days)).reshape(samples, -1)[:, :n]
    return values[positions].mean(axis=1)
