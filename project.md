# Forecasting the German day-ahead spot price

DBA3803 group project: the business case, the data, and how the forecast gets built.

---

## 1. The business case

A German industrial company runs **one 8-hour shift a day** and buys **500 MWh** for it,
62.5 MW in each hour of the shift. Electricity is a noticeable share of its production
cost, and so is staff, so when the shift runs is worth money.

Every day at **11:55 German time**, five minutes before the day-ahead auction closes at
12:00, it has to commit tomorrow's purchases, which means choosing when tomorrow's shift
starts, anywhere from 00:00 to 16:00. If it knew tomorrow's hourly prices, it would pick the
8 consecutive hours where electricity plus staff costs least. It does not know them, so it
forecasts them, and that forecast is this project.

The cost of a shift is electricity plus a staff premium:

| | |
|---|---|
| Electricity | the day-ahead price of each shift hour, times 62.5 MWh |
| Night and weekend premium | **625 EUR an hour** from 22:00 to 06:00, and every hour at the weekend, about 25% extra on a crew costing 2,500 EUR an hour. Spread over 62.5 MWh, that is 10 EUR/MWh |

Both are assumptions, stated in the report, and the premium is one constant in the notebook.
Choosing a shift means comparing sums of 8 prices, so the size of the price differences
matters, not only their order: a cheap hour is not worth it if it pulls expensive
neighbours into the shift. The premium is in euros, which a simple ranking of hours by
forecast wind and demand cannot weigh, but a price forecast can.

Three simplifications, all worth stating: a shift stays within one day, though a real shift
can cross midnight; the weekend premium covers every hour, so it raises the cost of a
weekend shift without changing which one wins; and night and weekend premiums do not stack,
though in practice a weekend night usually earns both.

We picked this use of a spot-price forecast on purpose. Richer ones exist, battery
arbitrage being the obvious example, but they add machinery without making anything
clearer. The value of a better forecast is easiest to see and defend when the only decision
is when to run.

### Two constraints on the decision

**It buys in Germany only.** The company's meter is in the DE-LU bidding zone, so that is
where it bids. A cheaper Swedish price is no use to it: trading on a price gap between
zones needs cross-border transmission capacity, which is allocated separately and held by
traders, and the gap itself is the congestion rent on the interconnector. The decision is
always when in the German day, never which country.

**It is a price taker.** Buying adds demand, which raises that hour's residual load and
moves the price up the supply stack. In the cheaper hours the supply curve slopes at
roughly 2.9 EUR/MWh per GW. At 500 MWh over an eight-hour shift, 62.5 MW an hour, the
company raises the price it pays by about 0.18 EUR/MWh, under 2% of what the forecast
saves it. The assumption only breaks at around 3,300 MW in a single hour, 6% of German
national demand, and no industrial buyer is that size. The report should still state the
assumption and give these numbers.

---

## 2. Notation: D, D-1, D-2

Power markets count days back from **D, the delivery day**, when the electricity flows.

| | Date in our example | What happens |
|---|---|---|
| **D-2** | 9 September | 12:00: the auction for 10 September clears |
| **D-1** | 10 September | **11:55: we run the model. 12:00: the auction for 11 September closes** |
| **D** | 11 September | The power flows, hour by hour. **This is what we predict** |

The auction for day D closes at noon on D-1, so the auction for D-1 closed at noon on D-2.
At 11:55 on D-1, every price for D-1 has been public for nearly 24 hours. That is why each
of the 24 target hours on day D already has a settled lag-24 value, and why step 3
predicts all 24 hours directly instead of walking forward through them.

## 3. What is predicted, and from when

The day-ahead auction is blind and closes at **12:00 German time on D-1**. One algorithm
then clears all 24 hours of day D at once. There is no second round and no revision: one
decision covers the whole day.

| | |
|---|---|
| We stand at | **10 September, 11:55 CEST** |
| We predict | **11 September, 00:00 to 23:00 CEST**, hour by hour |
| Output | a vector of 24 prices, EUR/MWh |
| Horizon | 12 to 36 hours ahead |

**Time zones.** Germany is UTC+2 in summer (CEST) and UTC+1 in winter (CET), so noon German
time is 10:00 UTC in September and 11:00 UTC in January: the same moment in market time, a
different hour in the data. Store everything in UTC and convert to German time only for
display. On the two clock-change days the output vector is 23 or 25 hours long, not 24.

---

## 4. What is knowable at 11:55

The project is built around this split. A model that uses anything from the second table
looks excellent in a backtest and is useless live.

**Knowable**

| | Why |
|---|---|
| TSO day-ahead forecasts of load, solar, onshore and offshore wind for day D | published ahead of delivery, see the caveat below |
| **All 24 cleared prices for day D-1** | D-1 cleared at noon on D-2, so the whole previous day is settled and public |
| Every price before that, back to 2019 | settled |
| Neighbouring zone prices for D-1 and earlier | same reason |
| The calendar | we know what day tomorrow is |

**Not knowable**

| | Why |
|---|---|
| Any actual generation or load on day D | measured after delivery |
| The day D price | it is the target |
| **Neighbouring zone prices for day D** | all coupled zones clear in the same auction at noon on D-1, so their day-D prices are decided at the same moment as ours |
| Actual generation on D-1 after about 11:00 | not yet metered and published |

Neighbour prices for day D are the easiest leak to miss, because they look like free
information.

### Caveat on the forecast data, for the report

The TSO day-ahead forecasts are computed at 08:00 on D-1 and published at 18:00 on D-1,
with the 12:00 gate between the two. Regulation (EU) 543/2013 Art 14.1(d) sets the 18:00
deadline, and TenneT states that it estimates wind at 08:00 for the following day and
publishes at 18:00.

So the numbers are made before the gate and published after it. That makes them:

1. Not hindsight: an 08:00 run knows nothing about the delivery day.
2. Not a revised series: the stored day-ahead series is a single vintage, with no later
   revisions written over it.
3. Unavailable at 11:55: nobody could have downloaded that file at the gate.

A real company would not use the TSO file. It would pay a forecast vendor for an equivalent
product available before the gate, an ordinary cost of doing this. Those forecasts are paid
and we have no access to them, so the TSO forecast stands in. The model is realistic in
substance even though this source is not available at 11:55. The limitations section says
so in one sentence, and it must not be left out.

---

## 5. The data

Three raw datasets from the same free API. `lib/fetching.py` only fetches them; the
notebook loads each with one call and does everything else.

```
lib/
  fetching.py          the API pull: fetch_de_prices, fetch_forecasts, fetch_neighbor_prices
  data_schemas.py      the data model the merged table is validated against
  splitting.py         the expanding-window folds
  modeling.py          Lasso-then-XGBoost, the asinh wrapper, columns -> model matrix
  evaluation.py        scoring and the fold loop
  decision.py          the shift choice and the staff premium
data/
  de_lu_day_ahead_prices.parquet       the target, 1 column               (~2 min to pull)
  de_lu_day_ahead_forecasts.parquet    the fundamentals, 4 columns        (~8 min)
  neighbor_prices.parquet              11 coupled bidding zones           (~25 min)
de_price_forecast.py                   parsing, exploration and modelling: edit this one
de_price_forecast.ipynb                its jupytext pair, synced from the .py, carries the outputs
```

Each function reads its file from `data/` if it exists and pulls it otherwise. Pass
`refresh=True` to pull again. No credentials needed.

### Dataset 1: the German target and its fundamentals

| Column | Unit | What |
|---|---|---|
| `price_da` | EUR/MWh | **the target.** DE-LU day-ahead clearing price |
| `fc_load` | MW | TSO day-ahead load forecast |
| `fc_solar` | MW | TSO day-ahead solar forecast |
| `fc_wind_onshore` | MW | TSO day-ahead onshore wind forecast |
| `fc_wind_offshore` | MW | TSO day-ahead offshore wind forecast |

Every column except the target is a forecast, never an actual. Actuals for day D are not
knowable at 11:55, so they are left out of the pull entirely instead of being filtered out
later.

### Dataset 2: the coupled bidding zones

`price_FR NL DK1 DK2 AT PL CH BE CZ SE4 NO2`, the day-ahead clearing price per zone.

Germany does not clear alone. One algorithm clears every coupled zone at once, subject to
transmission capacity, so a neighbour's settled price carries information about DE-LU that
nothing else here holds. Day D's neighbour prices are not knowable at the gate, because
those zones clear in the same auction as ours. D-1 and earlier only.

### Both

The index is `utc_timestamp`, timezone-aware UTC. The files are the raw pulls at whatever
resolution the API serves: no resampling, no joins, no lags, no calendar columns, no
scaling. All of that happens in the notebook.

**Source:** Energy-Charts API, Fraunhofer ISE, `api.energy-charts.info`, no key, CC BY 4.0.
Price series are relabelled from **Bundesnetzagentur | SMARD.de** and must be attributed to
them, not Fraunhofer, in the report.

**Quirks.** Forecasts are 15-minute throughout. Prices are hourly until the European
market-time-unit change on 2025-10-01 and 15-minute after it; the notebook averages both to
hourly. Prices range from -500 to +936 EUR/MWh, and -500 is the auction's technical floor,
not an outlier to clean. Negative hours rose from 211 in 2019 to 576 in 2025. The last rows
can carry a load forecast and no price, because the forecast runs ahead of the auction, so
drop unpriced rows before training. The API rate-limits hard, so the fetchers sleep 6
seconds between calls and back off on HTTP 429.

## 6. The three steps

### Step 1: data processing

Turn the 16 raw columns into a feature matrix where every column is knowable at 11:55.

| Group | Features | Why |
|---|---|---|
| Fundamentals | the four forecasts, and `fc_residual = fc_load - fc_solar - fc_wind_onshore - fc_wind_offshore` | Residual load is the most important variable: it is the demand the conventional fleet must cover, and the most expensive plant needed sets the price for everyone |
| Price memory | lags of 24, 48 and 168 hours; the previous day's mean, min and max; moving averages over a few windows | Lag 168 (same hour last week) captures the weekly shape. The moving averages stand in for the gas price, which sets the level and which we do not have |
| Neighbour memory | the 11 zones lagged 24 and 168 hours, and each zone's spread against DE-LU on D-1 | The spread shows whether the interconnector was binding, and in which direction |
| Calendar | hour of day, day of week, month, weekend flag, a smooth seasonality term | Taken from German local time, not UTC, because people's routines drive demand |

### The cutoff rule

**Never use anything published after 11:55 on D-1.**

This is not the same as "nothing within 24 hours of the target". The price for 23:00 on
D-1 sits one hour before target hour 00:00 on day D, but it cleared at noon on D-2, so it
is settled and legal. The whole of D-1 is available to every hour of day D.

We choose not to use it that way, and keep hourly lags same-hour: for target hour *h* on
day D, use hour *h* on D-1, D-2 and D-7. Lag 24 is informative because it sits at the same
point in the daily load and solar cycle, so pairing 23:00 with 23:00 means something, while
pairing 23:00 with 00:00 is a weak cross-hour relationship. The LEAR benchmark does feed all
24 prices of D-1 to every target hour; we decline that because the expected gain is small
and it doubles the feature count.

| Feature type | Structure |
|---|---|
| Hourly price lags | **same-hour**: lag 24, 48, 168 |
| Day-level aggregates | D-1's mean, min, max, spread; 7- and 30-day means ending at the close of D-1. **The same values attach to all 24 target hours of day D** |
| Neighbour prices | same-hour lag 24 and 168, plus each zone's spread against DE-LU on D-1 |
| Fundamentals | day D's own forecasts, which are known |
| Calendar | from German local time, not UTC |

Aggregates are the one place the whole of D-1 legitimately enters every hour, and they are
not a cross-hour relationship: "yesterday's level predicts tomorrow's level" applies to the
delivery day as a whole. The 30-day mean stands in for the gas price, which sets the level
of every hour and is absent from this dataset.

Anything built from actuals rather than forecasts must be lagged at least 168 hours,
because D-1 afternoon actuals are not published by 11:55.

Neighbour prices follow the same day rule: all coupled zones clear in the same auction, so
day D's neighbour prices are decided at the same moment as ours. D-1 and earlier only.

### Step 2: model fitting

Fit on train, choose hyperparameters on validation, and touch the test set once, at the
end.

If several methods land within a few percent of each other, that is a finding about the
problem, not a failure to tune, and reporting it is worth more than presenting only the
winner.

Evaluate with **MAE and RMSE**, plus a business metric: the euros the schedule saves. The
two can disagree, and when they do, the business case rests on the euros.

Compare every model with two baselines: **the price at the same hour the day before**, and
**the training-set mean**. A model that does not beat both is not worth using.

### Step 3: prediction, directly for all 24 hours

Produce the 24-hour vector for day D directly, all 24 hours at once, not recursively.

The obvious approach is to go hour by hour: predict 00:00, feed that prediction in as the
lag for 01:00, and walk forward. That is right for a typical multi-step time series and
wrong for this auction. At 11:55 on D-1, the whole of D-1 has already cleared, at noon on
D-2:

| Target hour on day D | lag-24 falls on | Known at 11:55? |
|---|---|---|
| 00:00 | D-1 00:00 | **yes** |
| 12:00 | D-1 12:00 | **yes** |
| 23:00 | D-1 23:00 | **yes** |

Every one of the 24 target hours already has a settled lag-24 value, and the same holds
for lag 48 and lag 168, so there is nothing to feed forward. Recursion would hurt in three
ways:

- Errors compound: hour 1's error enters hour 2's input, and by hour 23 the forecast rests
  on 22 layers of its own mistakes.
- It throws away information, putting a predicted price where a settled one exists.
- It changes the problem: each hour sees a different input distribution, so at prediction
  time the model is no longer doing what it was trained to do.

Measured on a stripped-down model, built so the two approaches differ only in their lag
structure, direct scores **21.42 MAE against 30.62 for recursive**. The breakdown by hour
of the delivery day shows where the gap comes from:

| Hour of day D | Recursive MAE | Direct MAE |
|---|---|---|
| 00:00 | **6.69** | 15.37 |
| 01:00 | **8.37** | 15.42 |
| 03:00 | **12.69** | 15.87 |
| 06:00 | 19.12 | **16.42** |
| 12:00 | 38.53 | **26.36** |
| 18:00 | 63.30 | **23.58** |
| 23:00 | 21.82 | **15.64** |

Recursion is better for the first three hours, because there its lag 1 is a settled price
(D-1 23:00). By 18:00 it has fed 18 of its own predictions back into itself and is almost
three times worse. The direct model stays flat across the day because every hour has an
equally good lag 24 behind it.

So: build the feature matrix for all 24 hours of day D, call `predict` once, get 24
numbers. This is **direct multi-horizon forecasting**, the standard in the electricity-price
forecasting literature, and it is both simpler and better.

Once the single model works, try fitting 24 separate models, one per delivery hour. Hour
03:00 and hour 19:00 behave differently, one set by baseload overnight and the other by the
evening peak. Per-hour models are standard in the field and untested here.

---

## 7. What done looks like

1. A feature pipeline where every column passes the 11:55 test, and a chronological split.
2. The chosen method against both baselines, on MAE and RMSE.
3. For one chosen day, the 24-hour predicted price curve against what actually cleared.
4. The euro answer: given the forecast, which shift would the company have run, what would
   it have paid for electricity and premiums, and how does that compare with a standard
   06:00 to 14:00 shift, with picking the shift by forecast residual load, and with perfect
   hindsight.
5. The limitations stated plainly, the forecast publication timing first among them.

Item 4 makes this a business project rather than a modelling exercise, and item 5 keeps it
honest.
