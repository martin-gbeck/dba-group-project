"""Figure 5 and Figure E1 of the group report (v4), redrawn from the repo, plus every
number the report's sections 7, 8 and appendix E take from the two notebooks, printed
unrounded.

Runs `de_price_forecast.py` (for `permutation_importance`) and `week_ahead.py` (for the
section 8 schedules), so it takes about as long as both notebooks. The figures keep the
look of the v3 images they replace: plotnine's minimal theme, 10 by 4.5 inches at 200 dpi.

Run from the project root:
`PYTHONPATH=. uv run python report/group/source/build_v4_figures.py`.
"""

import runpy
from pathlib import Path

import polars as pl
from plotnine import (
    aes,
    coord_flip,
    element_text,
    expand_limits,
    geom_col,
    geom_text,
    ggplot,
    labs,
    scale_x_discrete,
    theme,
    theme_minimal,
)

OUT = Path("report/group/figures")
OUT.mkdir(parents=True, exist_ok=True)
STYLE = theme_minimal() + theme(
    figure_size=(10, 4.5),
    legend_position="top",
    text=element_text(family="DejaVu Sans"),
)
pl.Config.set_tbl_rows(50)
pl.Config.set_float_precision(6)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(250)

week = runpy.run_path("week_ahead.py")
schedules: pl.DataFrame = week["schedules"]
print(schedules.sort("schedule", "year"))
print(
    schedules.filter(pl.col("year").is_in(["2023", "2024"]))
    .group_by("schedule", maintain_order=True)
    .agg(
        validation=pl.col("saving_share").mean(),
        fold_sd=pl.col("saving_share").std(),
    )
)

# Figure E1: what each schedule saves in 2025, in the v3 figure's order and names
E1_ROWS = {
    "daily-xgboost": ("daily, model (section 7)", "daily"),
    "daily-xgboost-rest": ("daily, model + rest rule", "daily"),
    "weekly-last-week-rest": ("weekly, last-week", "weekly"),
    "weekly-week-ahead-model-rest": ("weekly, week-ahead-model", "weekly"),
    "weekly-day-ahead-model-rest": ("weekly, dayahead-forecast*", "weekly"),
    "weekly-hindsight-rest": ("weekly, hindsight", "weekly"),
}
e1 = (
    schedules.filter(
        (pl.col("year") == "2025") & pl.col("schedule").is_in(list(E1_ROWS))
    )
    .with_columns(
        name=pl.col("schedule").replace_strict({k: v[0] for k, v in E1_ROWS.items()}),
        kind=pl.col("schedule").replace_strict({k: v[1] for k, v in E1_ROWS.items()}),
        millions=pl.col("saving_eur_per_year") / 1e6,
    )
    .with_columns(
        label=pl.format(
            "{}M ({}%)",
            pl.col("millions").round(2).cast(pl.String).str.pad_end(4, "0"),
            (pl.col("saving_share") * 100).round(1).cast(pl.String),
        )
    )
)
order = [v[0] for v in E1_ROWS.values()]
figure_e1 = (
    ggplot(e1, aes("name", "millions", fill="kind"))
    + geom_col()
    + geom_text(aes(label="label"), ha="left", nudge_y=0.05, size=8)
    + coord_flip()
    + scale_x_discrete(limits=order[::-1])
    + expand_limits(y=5.3)
    + labs(
        x="",
        y="saved against the standard 06:00 shift, million EUR a year",
        fill="",
        title="2025: what each schedule saves, 500 MWh a day (* not available on Sunday)",
    )
    + STYLE
)
figure_e1.save(OUT / "figE1-schedules-2025.png", dpi=200, verbose=False)
print(e1.select("name", "label"))

main = runpy.run_path("de_price_forecast.py")
importance: pl.DataFrame = main["importance"]
print(importance)
figure_5 = (
    ggplot(
        importance,
        aes("reorder(column, saving_share_lost)", "saving_share_lost", fill="feature_group"),
    )
    + geom_col()
    + geom_text(
        aes(label="saving_share_lost"), format_string="{:.1f}", ha="left", nudge_y=1, size=8
    )
    + coord_flip()
    + labs(
        x="",
        y="saving share lost when the column is shuffled, percentage points",
        fill="",
        title="What the final model relies on, 2023 and 2024",
    )
    + STYLE
)
figure_5.save(OUT / "fig5-permutation-importance.png", dpi=200, verbose=False)
