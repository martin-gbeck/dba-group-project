import polars as pl


def to_hourly(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Read one raw dataset, check it, and average it to hourly.

    The raw files are hourly until the European market time unit moved to 15 minutes in
    late 2025, and quarter-hourly after. The mean over each hour is the right reduction
    for both kinds of column: for price it is the hour's average clearing price, and for
    the forecasts it is average power over the hour, which is the hour's energy.

    It also repairs Switzerland for free. CH publishes hourly only, so three of every
    four quarter-hour rows are null, and a mean that skips nulls returns the one real
    value rather than propagating the gap.
    """
    return (
        lf.sort("utc_timestamp")
        .group_by_dynamic("utc_timestamp", every="1h")
        .agg(pl.all().mean())
    )
