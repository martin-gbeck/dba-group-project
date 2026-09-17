"""Schemas for the three raw datasets.

`strict = False` everywhere, so columns not described here pass through unvalidated.
Only the columns the project actually models on are listed.

`coerce = False` is deliberate, and it is about the timestamp. Under coercion pandera
does not reject a naive datetime column, it stamps UTC onto the wall-clock time:
`2023-11-14 22:13:20` becomes `2023-11-14 22:13:20+00:00`. Hand it a German-local column
and every row is silently wrong by an hour or two, in a project whose whole premise is a
noon cutoff. A per-field `coerce=False` does not override the Config, so the switch has
to be off at schema level. The fetchers already write correct dtypes, so coercion was
buying nothing and costing the one guarantee that matters.

The bounds are market facts rather than guesses. Day-ahead prices are bounded by the
exchange's technical limits: a floor of -500 EUR/MWh that gets hit exactly and often, so
it is real data and not an outlier to clean, and a ceiling far above anything in this
sample (the highest observed is 2,988 in France during the 2022 crisis).
"""

from typing import Annotated

import pandera.polars as pa
import polars as pl

UtcTimestamp = Annotated[pl.Datetime, "us", "UTC"]

PRICE_FLOOR = -500.0
PRICE_CEILING = 4000.0


class PriceSchema(pa.DataFrameModel):
    """DE-LU day-ahead clearing price. The target."""

    utc_timestamp: UtcTimestamp = pa.Field(unique=True, nullable=False)
    price_da: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=False)

    class Config:
        strict = False
        coerce = False


class ForecastSchema(pa.DataFrameModel):
    """TSO day-ahead forecasts for Germany, MW.

    Load carries a positive lower bound because national demand never reaches zero. The
    generation forecasts only need to be non-negative, since solar is zero every night.
    """

    utc_timestamp: UtcTimestamp = pa.Field(unique=True, nullable=False)
    fc_load: float = pa.Field(gt=0, le=120_000, nullable=False)
    fc_solar: float = pa.Field(ge=0, le=150_000, nullable=False)
    fc_wind_onshore: float = pa.Field(ge=0, le=150_000, nullable=False)
    fc_wind_offshore: float = pa.Field(ge=0, le=50_000, nullable=False)

    class Config:
        strict = False
        coerce = False


class NeighborPriceSchema(pa.DataFrameModel):
    """Day-ahead clearing prices for the bidding zones coupled to DE-LU.

    Every price column is nullable, and that is a property of the data rather than a
    defect. Switzerland publishes hourly only, so once Germany moved to 15-minute
    settlement in late 2025 its three intermediate quarters each hour are empty, 27% of
    the column. The Netherlands has a smaller scattering of gaps. Whatever consumes this
    has to choose between forward-filling and resampling down, so the schema states the
    nullability rather than hiding it.
    """

    utc_timestamp: UtcTimestamp = pa.Field(unique=True, nullable=False)
    price_FR: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)
    price_NL: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)
    price_DK1: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)
    price_DK2: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)
    price_AT: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)
    price_PL: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)
    price_CH: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)
    price_BE: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)
    price_CZ: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)
    price_SE4: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)
    price_NO2: float = pa.Field(ge=PRICE_FLOOR, le=PRICE_CEILING, nullable=True)

    class Config:
        strict = False
        coerce = False
