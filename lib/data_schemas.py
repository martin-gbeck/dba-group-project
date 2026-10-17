"""The data model for the merged hourly table, validated in the notebook right after the join.

`coerce=True` casts each column to the dtype stated here. `strict="filter"` drops every
column the model does not list, so the validated frame holds exactly these columns.

The bounds come from the market, not from the data. Day-ahead prices are bounded by the
exchange's technical limits: a floor of -500 EUR/MWh, which the auction does reach and is
not an outlier, and a ceiling of 4,000. Residual load goes negative when wind and solar
exceed demand, so it has no floor at zero.

Coercion never moves the timestamp: it arrives as UTC-aware datetime, so the cast is a
no-op. Coercing a naive datetime would stamp UTC onto wall-clock time without converting,
which is why the merge works in UTC from the start.
"""

from typing import Annotated

import pandera.polars as pa
import polars as pl
from pandera.api.polars.model_config import BaseConfig
from pandera.polars import PolarsData

# the datetime column must be in timezone UTC
UtcTimestamp = Annotated[pl.Datetime, "us", "UTC"]

PRICE_FLOOR = -500.0
PRICE_CEILING = 4000.0


class CombinedDataSchema(pa.DataFrameModel):
    """One row per delivery hour: the target, its fundamentals, the neighbours, the calendar."""

    class Config(BaseConfig):  # ty: ignore[invalid-mutable-override]
        """Cast every column to its declared type and drop the columns the model does not list."""

        coerce = True
        strict = "filter"

    # fmt: off
    utc_timestamp: UtcTimestamp = pa.Field(unique=True)                           # start of the delivery hour, UTC
    date: pl.Date = pa.Field()                                                    # delivery day, German local time
    year: pl.Int32 = pa.Field(ge=2019)                                            # year of the delivery day, local
    hour: pl.Int8 = pa.Field(ge=0, le=23)                                         # hour of day, German local time
    weekday: pl.Int8 = pa.Field(ge=1, le=7)                                       # 1 = Monday, 7 = Sunday

    price_da: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING)                  # target: DE-LU day-ahead price, EUR/MWh

    fc_load: float = pa.Field(gt=0, le=120_000)                                   # TSO day-ahead load forecast, MW
    fc_solar: float = pa.Field(ge=0, le=150_000)                                  # day-ahead solar forecast, MW, zero at night
    fc_wind_onshore: float = pa.Field(ge=0, le=150_000)                           # day-ahead onshore wind forecast, MW
    fc_wind_offshore: float = pa.Field(ge=0, le=50_000)                           # day-ahead offshore wind forecast, MW
    fc_residual: float = pa.Field(ge=-150_000, le=120_000)                        # load minus solar and wind, MW

    price_FR: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)   # France, same auction: lag 24h+ only
    price_NL: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)   # Netherlands, has a missing day
    price_DK1: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)  # West Denmark
    price_DK2: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)  # East Denmark
    price_AT: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)   # Austria
    price_PL: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)   # Poland
    price_CH: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)   # Switzerland, hourly only after 2025-10-01
    price_BE: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)   # Belgium
    price_CZ: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)   # Czech Republic
    price_SE4: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)  # South Sweden
    price_NO2: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)  # South Norway
    # fmt: on

    @pa.check("utc_timestamp")
    def hourly_grid(cls, data: PolarsData) -> pl.LazyFrame:
        """Consecutive rows are exactly one hour apart, so no hour is missing or repeated."""
        lf, column = data
        step = pl.col(column).diff()
        return lf.select(step.is_null() | (step == pl.duration(hours=1)))

    @pa.dataframe_check
    def residual_is_load_minus_renewables(cls, data: PolarsData) -> pl.LazyFrame:
        """fc_residual is derived from four other columns, so it is checked across them."""
        lf, _ = data
        renewables = pl.sum_horizontal(
            "fc_solar", "fc_wind_onshore", "fc_wind_offshore"
        )
        return lf.select(
            pl.col("fc_residual").is_close(
                pl.col("fc_load") - renewables,
                abs_tol=1e-6,
            )
        )
