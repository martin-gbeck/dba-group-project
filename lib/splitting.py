"""The expanding-window folds: each validation year is predicted from every year before it."""

import polars as pl


def make_folds(
    data: pl.DataFrame, folds: dict[str, int]
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    """folds: fold name to validation year, such as {"fold-1": 2023}.

    Returns fold name to (training rows, validation rows): training is every year before
    the validation year, validation is that year alone.
    """
    return {
        name: (data.filter(pl.col("year") < year), data.filter(pl.col("year") == year))
        for name, year in folds.items()
    }
