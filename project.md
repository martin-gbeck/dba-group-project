# Forecasting the German day-ahead spot price

DBA3803 group project. Everything about what this is and how it gets built.

---

## 1. The business case

**A German industrial company that produces something and needs electricity to do it.**

It knows it will consume, say, **500 MWh tomorrow**. It does not much care *when* during
the day it runs, as long as the total gets done. Electricity is a real share of its cost
of production, so the hours it picks are worth money.

Every day at **11:55 German time**, before the day-ahead auction closes at 12:00, it has
to commit tomorrow's purchases. If it knew tomorrow's hourly prices it would buy in the
cheapest hours and idle in the dearest, for the same total volume and a smaller bill.

It does not know them. So it forecasts them. **That forecast is this project.**

This is deliberately the simplest honest use of a spot-price forecast. There are richer
ones, battery arbitrage being the obvious example, but they add machinery without adding
clarity, and the value of a better forecast is easiest to see and to defend when the
decision is just "which hours do I buy in".

### Two constraints that define the decision

**It buys in Germany only.** The company's meter is in the **DE-LU bidding zone**, so that
is where it bids. Sweden clearing cheaper is not an opportunity available to it: capturing
a cross-zone price difference needs cross-border transmission capacity, which is allocated
separately and held by traders, and the price gap *is* the congestion rent on the
interconnector. The decision is always **which hours of the German day**, never which
country.

**It is a price taker.** Buying adds demand, which raises that hour's residual load and
walks the market up the supply stack. The supply curve slopes at roughly **2.9 EUR/MWh per
GW** in the cheaper hours. At 500 MWh spread over eight hours, 62.5 MW an hour, the company
pushes the price it pays up by about **0.18 EUR/MWh**, which is under 2% of what the
forecast saves it. The assumption only breaks around 3,300 MW in a single hour, which is
6% of German national demand. Nobody in industry is that size, so the assumption holds
comfortably and should be stated and quantified rather than assumed.

---

## 2. Notation: D, D-1, D-2

Standard in power markets. **D is the delivery day**, the day the electricity actually
flows. Everything else counts backwards from it.

| | Date in our example | What happens |
|---|---|---|
| **D-2** | 9 September | 12:00: the auction for 10 September clears |
| **D-1** | 10 September | **11:55: we run the model. 12:00: the auction for 11 September closes** |
| **D** | 11 September | The power flows, hour by hour. **This is what we predict** |

The chain this makes visible is the one the whole project turns on. The auction for day D
closes at noon on D-1, so the auction for D-1 closed at noon on **D-2**. Standing at 11:55
on D-1, every price for D-1 has been public for nearly 24 hours.

That is why each of the 24 target hours on day D already has a settled lag-24 value, and
therefore why step 3 predicts all 24 hours directly rather than walking forward through
them.

## 3. What is being predicted, and from exactly when

The day-ahead auction is **blind** and closes at **12:00 German time on D-1**. One
algorithm then clears **all 24 hours of day D simultaneously**. There is no second chance
and no rolling revision: it is one decision covering a whole day.

| | |
|---|---|
| We stand at | **10 September, 11:55 CEST** |
| We predict | **11 September, 00:00 to 23:00 CEST**, hour by hour |
| Output | a vector of 24 prices, EUR/MWh |
| Horizon | 12 to 36 hours ahead |

**The UTC trap.** Germany is UTC+2 in summer (CEST) and UTC+1 in winter (CET), so noon
German time is **10:00 UTC in September and 11:00 UTC in January**. Same instant in market
time, different hour in the data. Store everything in UTC, convert to German time only
when a human reads it. The output vector is **23 or 25 hours long on the two clock-change
days**, not always 24.

---

## 4. What is knowable at 11:55, and what is not

This is the constraint the whole project is built around. Get it wrong and the model looks
excellent and is worthless.

**Knowable**

| | Why |
|---|---|
| TSO day-ahead forecasts of load, solar, onshore and offshore wind for day D | published ahead of delivery, see the caveat below |
| **All 24 cleared prices for day D-1** | D-1 cleared at noon on **D-2**, so the whole previous day is settled and public |
| Every price before that, back to 2019 | settled |
| Neighbouring zone prices for D-1 and earlier | same reason |
| The calendar | we know what day tomorrow is |

**Not knowable**

| | Why |
|---|---|
| Any actual generation or load on day D | measured after delivery |
| The day D price, obviously | it is the target |
| **Neighbouring zone prices for day D** | all coupled zones clear in the *same* auction at noon on D-1, so their day-D prices are decided simultaneously with ours |
| Actual generation on D-1 after about 11:00 | not yet metered and published |

The neighbour one is the subtle trap. It looks like free information and it is a leak.

### The caveat on the forecast data, which goes in the report

The TSO day-ahead forecasts are **computed at 08:00 on D-1 and published at 18:00 on D-1**.
The gate is at 12:00, between the two. Regulation (EU) 543/2013 Art 14.1(d) sets the 18:00
deadline; TenneT states it estimates wind at 8 a.m. for the following day and publishes at
6 p.m.

So the numbers are **made before the gate and published after it**. Three separate things,
worth keeping apart:

1. **Not hindsight.** An 08:00 run knows nothing about the delivery day.
2. **Not a revised series.** The stored day-ahead series is a genuine single vintage, not
   later revisions written back over it.
3. **It is a point-in-time availability failure.** At 11:55 you could not have downloaded
   that file.

In practice a real company would not be using the TSO file. It would pay a forecast vendor
for an equivalent product available before the gate, which is an ordinary cost of doing
this. The model is therefore realistic in substance even though this particular source is
not strictly available at 11:55. **State it in the limitations section in one sentence.**
Leaving it unstated is the only unacceptable option.

---

## 5. The data

Two raw datasets, one fetch script each, both pulled from the same free API.

```
datafetch/
  energy_charts.py            shared: throttled GET, payload -> polars, hourly resample
  fetch_de_day_ahead.py       -> raw_data/de_lu_day_ahead_forecasts.parquet   (~7 min)
  fetch_neighbour_prices.py   -> raw_data/neighbour_prices.parquet            (~10 min)
raw_data/
  de_lu_day_ahead_forecasts.parquet    the target and its fundamentals, 5 columns
  neighbour_prices.parquet             11 coupled bidding zones, 11 columns
```

Both run with `uv run python datafetch/<script>.py` and need no credentials.

### Dataset 1 — the German target and its fundamentals

| Column | Unit | What |
|---|---|---|
| `price_da` | EUR/MWh | **the target.** DE-LU day-ahead clearing price |
| `fc_load` | MW | TSO day-ahead load forecast |
| `fc_solar` | MW | TSO day-ahead solar forecast |
| `fc_wind_onshore` | MW | TSO day-ahead onshore wind forecast |
| `fc_wind_offshore` | MW | TSO day-ahead offshore wind forecast |

Everything except the target is a **forecast**, never an actual. Actuals for day D are not
knowable at 11:55, so they are deliberately absent rather than left in to be filtered out
later. A model that reaches for them looks excellent and is worthless.

### Dataset 2 — the coupled bidding zones

`price_FR NL DK1 DK2 AT PL CH BE CZ SE4 NO2`, day-ahead clearing price per zone.

They matter because Germany does not clear alone: one algorithm clears every coupled zone
simultaneously subject to transmission capacity, so a neighbour's settled price carries
information about DE-LU that nothing else here holds. **And day D's neighbour prices are
not knowable at the gate**, because those zones clear in the same auction as ours. D-1 and
earlier only.

### Both

Index is `utc_timestamp`, timezone-aware UTC, hourly. Joining them and building features is
step 1 and is not done here: these files are the raw pulls, resampled to hourly and nothing
else. No lags, no aggregates, no calendar columns, no scaling.

**Source:** Energy-Charts API, Fraunhofer ISE, `api.energy-charts.info`, no key, CC BY 4.0.
Price series are relabelled from **Bundesnetzagentur | SMARD.de** and must be attributed to
them, not Fraunhofer, in the report.

**Quirks to know before you touch it.** Forecasts are natively 15-minute; price is hourly
until the European market-time-unit change in late 2025 and 15-minute after, so both are
mean-resampled to hourly. Price range is **-500 to +936 EUR/MWh**, and the -500 floor is
the auction's technical minimum, not an outlier to clean. Negative hours rose from 211 in
2019 to 576 in 2025. The last rows carry a load forecast and no price, because the forecast
runs ahead of the auction; drop unpriced rows before training. The API rate-limits hard, so
the fetchers sleep 6 seconds between calls and back off on 429.

## 6. The three steps

### Step 1 — Data processing

Turn the 16 raw columns into a feature matrix where **every column is knowable at 11:55**.

What goes in:

- **Fundamentals.** The four forecasts, and `fc_residual = fc_load - fc_solar -
  fc_wind_onshore - fc_wind_offshore`. Residual load is the single most important
  variable: it is the demand the conventional fleet must cover, and the most expensive
  unit needed sets the price for everyone.
- **Price memory.** Lags of **24, 48 and 168 hours**, the previous day's mean, min and max,
  and moving averages over a few windows. Same-hour-last-week (168) captures weekly shape;
  the moving averages act as a stand-in for the gas price, which is the thing actually
  setting the level and which we do not have.
- **Neighbour memory.** The 11 zones lagged 24 and 168 hours, and each one's spread against
  DE-LU yesterday, which reads on whether the interconnector was binding and which way.
- **Calendar.** Hour of day, day of week, month, weekend flag, and a smooth
  seasonality term. Derive hour of day from **German local time**, not UTC, because it is
  human behaviour that drives demand.

### The cutoff rule

**Never use anything published after 11:55 on D-1.**

That is the real constraint, and it is not quite the same as "nothing within 24 hours of
the target". The price for 23:00 on D-1 sits one hour before target hour 00:00 on day D,
but it cleared at noon on **D-2**, so it is settled and legal. The whole of D-1 is
technically available to every hour of day D.

**We choose not to use it that way.** Hourly lags stay **same-hour**: for target hour *h*
on day D, use hour *h* on D-1, D-2 and D-7. The reason is that lag-24 is informative
because it sits at the same point in the daily load and solar cycle, so pairing 23:00 with
23:00 is meaningful while pairing 23:00 with 00:00 is a weak cross-hour relationship. The
LEAR benchmark does feed all 24 prices of D-1 to every target hour; we are declining that
for a small expected gain and a doubled feature count.

So, concretely:

| Feature type | Structure |
|---|---|
| Hourly price lags | **same-hour**: lag 24, 48, 168 |
| Day-level aggregates | D-1's mean, min, max, spread; 7- and 30-day means ending at the close of D-1. **The same values attach to all 24 target hours of day D** |
| Neighbour prices | same-hour lag 24 and 168, plus each zone's spread against DE-LU on D-1 |
| Fundamentals | day D's own forecasts, which are known |
| Calendar | from German local time, not UTC |

Aggregates are the one place the whole of D-1 legitimately enters every hour, and that is
not a cross-hour relationship: "yesterday's level predicts tomorrow's level" applies to the
delivery day as a whole. The 30-day mean is doing a specific job, standing in for the gas
price, which sets the level of the entire surface and is absent from this dataset.

**Anything built from actuals rather than forecasts must be lagged at least 168 hours**,
because D-1 afternoon actuals are not published at 11:55.

**Neighbour prices follow the same day rule**, and the reason is easy to miss: all coupled
zones clear in the same auction, so day D's neighbour prices are decided simultaneously
with ours. D-1 and earlier only.

### Step 2 — Model fitting

Fit on train, choose hyperparameters on validation, touch the test set once at the end.

The course covers linear methods, model selection, trees and neural networks, so fit one of
each and report the comparison. **A table where several methods land within a few percent
of each other is a result about the problem, not a failure to tune**, and saying so is
worth more than quietly presenting only the winner.

Evaluate with **MAE and RMSE**, and add a business metric: the euros the schedule actually
saves. They can disagree, and when they do the euro number is the one the business case
rests on.

Always include two baselines, because a model that cannot beat them has no story:
**yesterday's price at the same hour**, and **the mean of the training set**.

### Step 3 — Prediction, and this is where the interesting design choice is

Produce the 24-hour vector for day D.

**Do it directly, not recursively. All 24 hours at once, in parallel.**

Measured, on a stripped-down model built so the two approaches differ only in their lag
structure: **direct 21.42 MAE, recursive 30.62**. The average hides the real story, which
is in the breakdown by hour of the delivery day:

| Hour of day D | Recursive MAE | Direct MAE |
|---|---|---|
| 00:00 | **6.69** | 15.37 |
| 01:00 | **8.37** | 15.42 |
| 03:00 | **12.69** | 15.87 |
| 06:00 | 19.12 | **16.42** |
| 12:00 | 38.53 | **26.36** |
| 18:00 | 63.30 | **23.58** |
| 23:00 | 21.82 | **15.64** |

Recursion is genuinely **better for the first three hours**, because there its lag-1 is a
real settled price (D-1 23:00). By 18:00 it has fed eighteen of its own predictions back
into itself and is nearly three times worse. That is error compounding drawn as a curve.
The direct model is flat across the day because every hour has an equally good lag-24
behind it.

The instinct is to go hour by hour: predict 00:00, then feed that prediction in as the lag
when predicting 01:00, and walk forward. That is the right approach for a genuine
multi-step time series problem. **It is wrong here, for a reason specific to how this
auction works.**

At 11:55 on D-1, the whole of D-1 has already cleared, at noon on D-2. So:

| Target hour on day D | lag-24 falls on | Known at 11:55? |
|---|---|---|
| 00:00 | D-1 00:00 | **yes** |
| 12:00 | D-1 12:00 | **yes** |
| 23:00 | D-1 23:00 | **yes** |

**Every one of the 24 target hours already has a real, settled lag-24 value.** There is no
gap to fill and therefore nothing to feed forward. The same holds for lag-48 and lag-168.

Recursion would be actively harmful:

- **Errors compound.** Hour 1's error enters hour 2's input, and by hour 23 you are
  forecasting off twenty-two layers of your own mistakes.
- **It throws away better information.** You would be substituting a predicted price where
  a settled one exists.
- **It changes the problem.** Each hour would face a different input distribution, so the
  model is no longer doing at prediction time what it was trained to do.

So: build the feature matrix for all 24 hours of day D, call `predict` once, get 24 numbers.
This is called **direct multi-horizon forecasting**, it is what the electricity price
forecasting literature does, and it is simpler as well as better.

One refinement worth trying once the single model works: **fit 24 separate models, one per
delivery hour.** Hour 03:00 and hour 19:00 are genuinely different regimes, one set by
baseload overnight and the other by the evening peak. This is standard practice in the
field and it is untested here.

---

## 7. What done looks like

1. A feature pipeline where every column passes the 11:55 test, and a chronological split.
2. A comparison across the model families on the syllabus, with both baselines, on MAE and
   RMSE.
3. For one chosen day, the 24-hour predicted price curve against what actually cleared.
4. The euro answer: given the forecast, which hours would the company have bought in, what
   would it have paid, and how does that compare with buying on a fixed schedule and with
   perfect hindsight.
5. The limitations stated plainly, the forecast publication timing first among them.

Item 4 is the one that makes it a business project rather than a modelling exercise. Item 5
is the one that makes it honest.
