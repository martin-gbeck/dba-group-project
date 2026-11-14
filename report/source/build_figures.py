"""The report's figures, rebuilt from the project code.

Runs `week_ahead.py`, which fits the final model (its day-ahead row is the main
notebook's final model, scored identically on 2025) and the week-ahead model, then
draws each figure from its results into `report/figures/`.

Run from the project root: `PYTHONPATH=. uv run python report/source/build_figures.py`.
"""

import runpy
from pathlib import Path

import polars as pl
from plotnine import (
    aes,
    annotate,
    coord_flip,
    facet_wrap,
    geom_col,
    geom_hline,
    geom_line,
    geom_point,
    geom_rect,
    geom_text,
    ggplot,
    labs,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
)

from lib.decision import best_shift, daily_costs, premium

FIGURES = Path("report/figures")
FIGURES.mkdir(parents=True, exist_ok=True)
GREY = "#9a9a9a"
EMPHASIS = "#1f77b4"
BASE = theme_minimal() + theme(figure_size=(8, 3.6), legend_position="none")

notebook = runpy.run_path("week_ahead.py")
df: pl.DataFrame = notebook["df"]
test_rest_frame: pl.DataFrame = notebook["test_rest_frame"]
flex_2025: pl.DataFrame = notebook["flex_2025"]

# Figure 1: where the cheapest shift starts, 2019 to 2023 (the exploration years)
explored = df.filter(pl.col("year") <= 2023)
cheapest_starts = (
    explored.filter(best_shift(pl.col("price_da") + premium()))
    .group_by("date")
    .agg(start=pl.col("hour").first())
    .group_by("start")
    .agg(days=pl.len())
    .with_columns(share=pl.col("days") / pl.col("days").sum())
)
all_starts = pl.DataFrame({"start": list(range(17))}, schema={"start": pl.Int8})
cheapest_starts = (
    all_starts.join(cheapest_starts.with_columns(pl.col("start").cast(pl.Int8)), on="start", how="left")
    .fill_null(0)
    .with_columns(
        kind=pl.when(pl.col("start") == 6).then(pl.lit("standard")).otherwise(pl.lit("other"))
    )
)
standard_share = cheapest_starts.filter(pl.col("start") == 6)["share"].item()
figure_1 = (
    ggplot(cheapest_starts, aes("start", "share", fill="kind"))
    + geom_col()
    + annotate(
        "text",
        x=6,
        y=0.035,
        label=f"standard 06:00 start: {standard_share:.0%} of days",
        size=8,
    )
    + scale_fill_manual(values={"standard": EMPHASIS, "other": GREY})
    + scale_x_continuous(breaks=list(range(0, 17, 2)))
    + scale_y_continuous(labels=lambda values: [f"{v:.0%}" for v in values])
    + labs(
        x="start hour of the cheapest shift, German time",
        y="share of days",
        title="The cheapest shift starts at midnight or mid-morning, almost never at 06:00",
    )
    + BASE
)
figure_1.save(FIGURES / "fig1-cheapest-start.png", dpi=200, verbose=False)

# Figure 2: 2025 saving share by way of choosing the shift
methods = {
    "Final model (day-ahead)": notebook["test_scores"]["day-ahead"]["saving_share"],
    "Residual-load rule": None,  # filled from rest_2025 below, same 2025 scoring
    "Week-ahead model": notebook["test_scores"]["week-ahead"]["saving_share"],
    "Last year's profile": notebook["test_scores"]["last-year-profile"]["saving_share"],
}
rest_2025: pl.DataFrame = notebook["rest_2025"]
methods["Residual-load rule"] = rest_2025.filter(pl.col("schedule") == "residual-load-rule")[
    "saving_share"
].item()
shares = pl.DataFrame(
    {"method": list(methods), "saving_share": list(methods.values())}
).with_columns(
    kind=pl.when(pl.col("method").str.starts_with("Final")).then(pl.lit("final")).otherwise(pl.lit("other")),
    label=pl.col("saving_share").map_elements(lambda v: f"{v:.1%}", return_dtype=pl.String),
    method=pl.col("method").cast(pl.Enum(list(reversed(list(methods))))),
)
figure_2 = (
    ggplot(shares, aes("method", "saving_share", fill="kind"))
    + geom_col(width=0.6)
    + geom_text(aes(label="label"), ha="left", nudge_y=0.01, size=9)
    + coord_flip()
    + scale_fill_manual(values={"final": EMPHASIS, "other": GREY})
    + scale_y_continuous(limits=(0, 1.1), labels=lambda values: [f"{v:.0%}" for v in values])
    + labs(
        x="",
        y="share of the possible saving against the standard 06:00 shift, 2025",
        title="The forecast's shift gets 95.9% of the possible saving, ahead of the rule",
    )
    + BASE
    + theme(figure_size=(8, 2.8))
)
figure_2.save(FIGURES / "fig2-saving-share-2025.png", dpi=200, verbose=False)

# Figure 3: the model's running lead over the residual-load rule through 2025
day_costs = test_rest_frame.rename({"day_ahead": "pred"}).pipe(daily_costs)
lead = day_costs.select("date", lead=(pl.col("rule") - pl.col("model")).cum_sum())
final_lead = lead["lead"][-1]
figure_3 = (
    ggplot(lead, aes("date", "lead"))
    + geom_hline(yintercept=0, color=GREY)
    + geom_line(color=EMPHASIS)
    + annotate(
        "text",
        x=lead["date"][-1],
        y=final_lead,
        label=f"{final_lead / 1000:,.0f}k EUR",
        ha="right",
        va="bottom",
        size=8,
    )
    + scale_y_continuous(labels=lambda values: [f"{v / 1000:,.0f}k" for v in values])
    + labs(
        x="",
        y="model's saving minus the rule's, EUR",
        title="The lead over the residual-load rule builds up over many days of 2025, not one spell",
    )
    + BASE
)
figure_3.save(FIGURES / "fig3-running-lead-2025.png", dpi=200, verbose=False)

# Figure 4: one typical day, 6 March 2025 (the weekday closest to 2025's median day)
daily_mean = test_rest_frame.group_by("date").agg(
    mean_price=pl.col("price_da").mean(), weekday=pl.col("weekday").first()
)
median_day = daily_mean["mean_price"].median()
example_date = (
    daily_mean.filter(pl.col("weekday") <= 5)
    .sort((pl.col("mean_price") - median_day).abs())
    .row(0, named=True)["date"]
)
day = test_rest_frame.filter(pl.col("date") == example_date).with_columns(
    cleared=pl.col("price_da") + premium(),
    forecast=pl.col("day_ahead") + premium(),
    picked=best_shift(pl.col("day_ahead") + premium()),
)
picked_hours = day.filter(pl.col("picked"))["hour"].to_list()
lines = day.unpivot(index="hour", on=["cleared", "forecast"], variable_name="series", value_name="eur")
band = pl.DataFrame({"xmin": [min(picked_hours) - 0.5], "xmax": [max(picked_hours) + 0.5]})
figure_4 = (
    ggplot(lines, aes("hour", "eur"))
    + geom_rect(
        data=band,
        mapping=aes(xmin="xmin", xmax="xmax"),
        ymin=-10,
        ymax=240,
        inherit_aes=False,
        fill=EMPHASIS,
        alpha=0.12,
    )
    + geom_line(aes(color="series"))
    + geom_point(aes(color="series"), size=1)
    + geom_text(
        data=lines.filter(pl.col("hour") == 23),
        mapping=aes(label="series", color="series"),
        ha="left",
        nudge_x=0.3,
        size=8,
    )
    + annotate("text", x=min(picked_hours), y=232, label="model's shift = cheapest shift", ha="left", size=8)
    + scale_x_continuous(breaks=list(range(0, 24, 3)), limits=(0, 26))
    + labs(
        x="hour of day, German time",
        y="price plus premium, EUR/MWh",
        title=f"On a typical day the forecast is too high but picks the cheapest shift ({example_date:%d %b %Y})",
    )
    + BASE
)
figure_4.save(FIGURES / "fig4-one-day.png", dpi=200, verbose=False)

# Figure 5: plan the week, adjust the day before (week_ahead section 6)
setting_names = {False: "free choice", True: "11 hours of rest kept"}
setting_order = pl.Enum(list(setting_names.values()))
within_four = (pl.col("earlier") <= 4) & (pl.col("later") <= 4)
either_way = flex_2025.filter(within_four, pl.col("earlier") == pl.col("later")).with_columns(
    move=pl.lit("either way"), hours=pl.col("earlier")
)
earlier_only = flex_2025.filter(within_four, pl.col("later") == 0).with_columns(
    move=pl.lit("earlier only"), hours=pl.col("earlier")
)
flex_lines = pl.concat([either_way, earlier_only]).with_columns(
    setting=pl.col("keep_rest").replace_strict(setting_names, return_dtype=setting_order)
)
references = pl.concat(
    [
        flex_2025.filter(pl.col("reach") == "any-start").with_columns(level=pl.lit("day-ahead choice")),
        flex_2025.filter(pl.col("reach") == "either-way-0h").with_columns(level=pl.lit("week-ahead plan")),
    ]
).with_columns(setting=pl.col("keep_rest").replace_strict(setting_names, return_dtype=setting_order))
figure_5 = (
    ggplot(flex_lines, aes("hours", "saving_share"))
    + geom_hline(data=references, mapping=aes(yintercept="saving_share"), linetype="dashed", color=GREY)
    + geom_text(
        data=references,
        mapping=aes(x=2.5, y="saving_share", label="level"),
        va="bottom",
        size=7,
        color=GREY,
    )
    + geom_line(aes(color="move"))
    + geom_point(aes(color="move"))
    + geom_text(
        data=flex_lines.filter(pl.col("hours") == 4),
        mapping=aes(x=4.1, label="move", color="move"),
        ha="left",
        size=7,
    )
    + facet_wrap("setting")
    + scale_x_continuous(breaks=[0, 1, 2, 3, 4], limits=(0, 5.4))
    + labs(
        x="hours the day-ahead choice may move the planned start",
        y="share of the possible saving, 2025",
        title="A one-hour move wins back half of what a week's notice costs, and all of it with rest kept",
    )
    + BASE
    + theme(figure_size=(8, 3.8))
)
figure_5.save(FIGURES / "fig5-plan-then-adjust.png", dpi=200, verbose=False)
print("figures written to", FIGURES)
