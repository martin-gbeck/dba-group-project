"""Scoring: the price error, the cost of the shift a forecast picks, and the fold loop.

Every function works on the hourly table built in notebook section 4.2, which carries
`date`, `hour`, `weekday`, `price_da` and `price_lag24`.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, TypedDict, cast

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
    """Scores averaged over the validation years, with the saving share and MAE of each
    year kept, in fold order, for the paired comparisons."""

    saving_share_folds: list[float]  # one saving share per validation year
    mae_folds: list[Metric]  # one MAE per validation year


class Comparison(TypedDict):
    """Rule 1 between two options, from their gaps on each validation year."""

    saving_share_gap: float  # mean over the years of first minus second
    saving_share_gap_se: float  # SD of the yearly gaps / sqrt(number of years)
    mae_gap: Metric  # mean over the years of first minus second, EUR/MWh
    mae_gap_se: Metric
    winner: str  # the name of the option that leads, or "tie" on both counts


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

    Returns the averaged scores, with each year's saving share and MAE, and the
    out-of-fold predictions for diagnostics.
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
    """Average the validation years' scores and keep each year's saving share and MAE."""
    table = pl.DataFrame(scores)
    summary = {column: table[column].mean() for column in table.columns}
    summary["saving_share_folds"] = [score["saving_share"] for score in scores]
    summary["mae_folds"] = [score["mae"] for score in scores]
    # the table's columns are exactly the keys of Scores, so the averages plus the two
    # yearly lists make a CVScores
    return cast(CVScores, summary)


def paired_gap(first: list[float], second: list[float]) -> tuple[float, float]:
    """The mean of the yearly gaps first minus second, and its standard error, the SD of
    the gaps over the square root of the number of years."""
    gaps = np.array(first) - np.array(second)
    return float(gaps.mean()), float(gaps.std(ddof=1) / np.sqrt(len(gaps)))


def compare(
    first: Mapping[str, Any], second: Mapping[str, Any], names: tuple[str, str]
) -> Comparison:
    """Rule 1 without its last step: the option whose mean yearly gap in saving share is
    more than one standard error in its favour leads; within one standard error, the
    same test on MAE, where lower is better, decides. `first` and `second` carry
    `saving_share_folds` and `mae_folds`, as `cross_validate` returns them. An option
    without MAE, such as the residual-load rule, ties on MAE. The simpler option on a tie
    on both counts is left to the reader."""
    saving_gap, saving_se = paired_gap(
        first["saving_share_folds"], second["saving_share_folds"]
    )
    has_mae = None not in first["mae_folds"] + second["mae_folds"]
    mae_gap, mae_se = (
        paired_gap(first["mae_folds"], second["mae_folds"]) if has_mae else (None, None)
    )
    if saving_gap > saving_se:
        winner = names[0]
    elif -saving_gap > saving_se:
        winner = names[1]
    elif mae_gap is not None and mae_se is not None and -mae_gap > mae_se:
        winner = names[0]
    elif mae_gap is not None and mae_se is not None and mae_gap > mae_se:
        winner = names[1]
    else:
        winner = "tie"
    return Comparison(
        saving_share_gap=saving_gap,
        saving_share_gap_se=saving_se,
        mae_gap=mae_gap,
        mae_gap_se=mae_se,
        winner=winner,
    )


def best_setting(settings: pl.DataFrame) -> pl.DataFrame:
    """Rule 2: the one-row frame of the setting with the lowest MAE among those the top
    setting by saving share does not lead, by mean yearly gap beyond one standard error.
    `settings` holds one row per setting with the columns `cross_validate` returns."""
    top = settings.sort("saving_share", descending=True).row(0, named=True)
    within_reach = [
        paired_gap(top["saving_share_folds"], row["saving_share_folds"])
        for row in settings.iter_rows(named=True)
    ]
    ties_with_top = pl.Series([gap <= se for gap, se in within_reach])
    return settings.filter(ties_with_top).sort("mae").head(1)


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
