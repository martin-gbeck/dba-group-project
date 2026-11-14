# Fact sheet: DBA3803 group report

Every number the report may use, with where it comes from. "Main [n]" is cell n of `de_price_forecast.ipynb` (0-based, as run on 2026-11-14); "Week [n]" is cell n of `week_ahead.ipynb` (run 2026-11-14, commit 11099dc). Figures are built by `report/source/build_figures.py`, which runs `week_ahead.py`. No number goes in the report unless it is here.

**Rounding, fixed once:** saving shares to one decimal in % (95.9%); EUR a year to the thousand in text (83,000) or to two decimals in millions (4.29 million); MAE to one decimal in EUR/MWh. **Year, fixed once:** the test result is always 2025; validation is always the mean of 2023 and 2024 ("validation years"); the reserve is always "2026, 1 January to 11 October".

## 1. The business case (project.md section 1; `lib/decision.py`)

| Fact | Value | Source |
|---|---|---|
| Shifts a day | one, 8 hours | project.md 1; `SHIFT_HOURS = 8` |
| Energy per shift | 500 MWh, 62.5 MW in each hour | `SHIFT_MW = 62.5` |
| Decision time | 11:55 German time on D-1, five minutes before the day-ahead auction closes at 12:00 | project.md 1; source card [9] |
| Allowed start hours | 00:00 to 16:00, 17 options; shift stays within one day | project.md 1; Main [84] |
| Staff premium | 625 EUR an hour from 22:00 to 06:00 and every weekend hour, about 25% on a crew costing 2,500 EUR an hour; = 10 EUR/MWh over 62.5 MWh | `PREMIUM_EUR_PER_HOUR = 625.0`; project.md 1 |
| Night and weekend premiums | do not stack (simplification) | project.md 1 |
| Standard shift | 06:00 to 14:00 every day | `STANDARD_SHIFT = (6, 13)` |
| Price taker | slope of price on residual load in 2025's below-median-price hours: 2.94 EUR/MWh per GW; 62.5 MW raises the price by about 0.18 EUR/MWh | computed 2026-11-14 with `scratchpad/slope.py` (cov/var on 2025 hours below the median price); project.md 1 says 2.9 and 0.18 |
| Price taker, against the saving | 0.18 EUR/MWh is under 1% of the 23.5 EUR/MWh the model's shift saves against the standard shift on 2025 (83.3 minus 59.8) | Main [191] shift_cost column |
| Price taker, break point | about 3.4 GW (10 EUR/MWh / 2.94), 6% of 2025's mean load forecast of 53.6 GW, would move the price by the size of the premium | computed 2026-11-14; project.md says "around 3,300 MW, 6%" (same order; use 3.4 GW) |

## 2. Metric definitions

- `saving_share = (standard - shift_cost) / (standard - hindsight)`, each the year's average cost in EUR/MWh, price plus premium, of: the standard 06:00 shift, the shift the method picks, the cheapest shift in hindsight. 100% = hindsight, 0% = standard shift. From the year's totals, so a day with more to save counts more. Main [107]; `lib/evaluation.py` `evaluate`.
- `saving_eur_per_year`: (standard cost - picked cost) x 62.5 MWh x 8 hours x 365. Main [189].
- `right_shift_share`: share of days the picked shift is the cheapest one. Main [107].
- MAE, RMSE in EUR/MWh; MASE = MAE / MAE of same hour yesterday. Main [107]; card [4].
- Paired fold gap rule: gap = A - B in each validation year; SE = SD of the yearly gaps / sqrt(2); A leads if mean gap > SE, which with two years means A beats B in both years. Main [107].
- Three selection rules: (1) between methods, lead on saving share wins, then MAE, then the simpler; (2) a method's best setting is the lowest MAE among settings that tie with the top setting on saving share; (3) a wider search's best replaces a chosen setting only if it wins under rule 1. Main [107].

## 3. Data (Main [14] to [31]; card [10])

| Fact | Value | Source |
|---|---|---|
| Source | Energy-Charts API (Fraunhofer ISE), three datasets: DE-LU day-ahead prices; TSO day-ahead forecasts of load, solar, onshore and offshore wind; prices of 11 neighbouring zones (FR, NL, DK1, DK2, AT, PL, CH, BE, CZ, SE4, NO2) | Main [3], [14], [62] |
| Span | 2019-01-01 to 2026-10-11 (German dates) | Main [23] |
| Raw rows | prices 95,255; forecasts 272,732; neighbours 95,255 | Main [23] |
| Quarter-hour prices | from 2025-10-01 German time; averaged to hours | Main [24], [98] |
| Duplicate timestamps | 0 in every file | Main [23] |
| Clock-change days | 8 with 23 hours, 7 with 25 hours | Main [26] |
| Missing values | only the Dutch price, 24 hours (all of 2026-10-10), forward-filled: every hour takes the last known Dutch price (23:00 on 9 Oct) | Main [29], [96] |
| Auction floor | -500 EUR/MWh hit once in train (2023) | Main [26], [72] |
| Residual load | load minus solar minus onshore wind minus offshore wind, built by us | Main [6] |
| Forecast timing | Load day-ahead forecast must be published at least 2 hours before day-ahead gate closure, so it is available at 11:55. Wind and solar day-ahead forecasts are due by 18:00 on D-1, after the gate | card [8] (Reg. 543/2013 Art. 6(2)(b), 14(2)(d)). **Correction:** project.md and Main [31] say all `fc_*` columns are published at 18:00; by law only wind and solar are. The limitation stands for wind, solar and residual load |
| What a buyer would do | buy a vendor forecast made the same morning; the TSO forecast stands in for it | Main [31] |
| Actuals | none in the data | Main [31] |

## 4. Split (Main [17], [19], [105])

| Set | Hours | Dates |
|---|---|---|
| Fold 1 train / validate | 35,064 / 8,760 | 2019-2022 / 2023 |
| Fold 2 train / validate | 43,824 / 8,784 | 2019-2023 / 2024 |
| Development (`dev`, after dropping the first 30 days) | 51,888 | 2019-01-31 to 2024-12-31 |
| Test | 8,760 | 2025-01-01 to 2025-12-31 |
| Reserve | 6,815 | 2026-01-01 to 2026-10-11 (284 days) |

Exploration used 2019 to 2023, so fold 1's score is slightly optimistic; 2024 was never explored. Main [17].

## 5. Exploration (training years 2019 to 2023)

| Fact | Value | Source |
|---|---|---|
| Yearly mean price | 2019 37.7, 2020 30.5, 2021 96.9, 2022 235.4, 2023 95.2 EUR/MWh; almost 8 times from 2020 to 2022 | Main [37] |
| Naive miss (same hour yesterday) | 9.3 EUR/MWh in 2019, 58.2 in 2022; always 25% to 33% of the year's mean | Main [37] |
| Same hour yesterday vs last week | yesterday beats last week in every year | Main [37] |
| Shape share (within-day variance / total) | 2019 0.418, 2020 0.367, 2021 0.175, 2022 0.194, 2023 0.431; so in 2021 and 2022 over 80% of movement was whole days moving together | Main [41] |
| Cheapest hours moved | from about 03:00 to about 13:00 to 14:00 by 2023 (solar) | Main [45], Main [44] figure |
| Residual load and price, within day | correlation 0.76 to 0.85 every year; pooled 0.70 | Main [51], [54] |
| Slope of price on residual load | 1.1 EUR/MWh per GW in 2019, 7.1 in 2022, 3.1 in 2023 | Main [51] |
| Scarcity bend | the 2021 price line bends upward above about 52 GW of residual load | Main [49] |
| Best single component within day | at most 0.29 (solar -0.288) | Main [54] |
| Autocorrelation | lag 24 h 0.90; lag 168 h 0.83 vs 0.81 at 144 and 0.80 at 192 | Main [59] |
| Neighbour lag 24 | best zones CZ 0.906, BE 0.904 only tie Germany's own 0.903 | Main [62], [63] |
| Negative hours | 1,018 over 2019-2023 (211, 298, 139, 69, 301); 2.3% of train hours; peak at 13:00 to 14:00 | Main [72], [93] |
| Weekend | Sunday 27 EUR/MWh below its week's mean, cheaper in 89% of weeks; Saturday 15 below | Main [76] |
| Cheapest shift start, 2019-2023 | 00:00 on 30% to 39% of days; 09:00 or 10:00 together on 40% to 57%; 06:00 on 1% at most (pooled 1%, Figure 1) | Main [88], [89]; fig1 |
| Simple rules, 2019-2023 | residual-load rule 87% to 97% of the possible saving; yesterday's prices 52% to 71%; last year's profile 26% to 54% | Main [91], [92] |
| Example, 2023 | cheapest 74.5, standard 97.4, rule's pick 76.2 EUR/MWh | Main [91] |
| Why the rule misses | ignores the premium; misses most in calm 2019, 2020 when 10 EUR/MWh is large next to price gaps | Main [92] |

## 6. Features (Main [98], [101], [167])

Five groups tested: calendar (hour, weekday, month, weekend flag); fundamentals (load, solar, onshore wind, offshore wind, residual load; the Lasso stage gets only residual load and solar); price memory (lags 24, 48, 168 h; D-1 mean, min, max; 7-day and 30-day mean of daily price); level interaction (residual load in GW x 30-day mean); neighbour spreads (zone minus Germany at the same hour on D-1, 11 zones). The final model has 17 columns: calendar 4, fundamentals 5, price memory 8. Every feature is known at 11:55 D-1 (apart from the forecast timing caveat in section 3).

## 7. Model comparison and selection (validation years, mean of 2023 and 2024)

### 7.1 Methods at default settings (Main [114], [117])

| Method | Saving share | By year | MAE | MASE |
|---|---|---|---|---|
| Lasso-then-XGBoost (default) | 95.6% | 94.7%, 96.6% | 17.5 | 0.636 |
| XGBoost alone (default) | 95.4% | 95.6%, 95.1% | 18.9 | 0.686 |
| Residual-load rule | 94.1% | 93.0%, 95.2% | n/a | n/a |
| LEAR | 93.5% | 92.7%, 94.4% | 17.1 | 0.621 |
| Same hour yesterday | 73.7% | | 27.5 | 1.000 |
| Last year's profile | 55.3% | | n/a | |
| Standard shift | 0% | | | |
| Training mean | -22.0% | | 36.6 | 1.330 |

- LEAR has the lowest MAE of the three methods but loses to both tree methods on saving share in both years (gaps 2.1 and 1.8 points) and to the residual-load rule (0.5 points). Main [118].
- Default trees tie with each other (gap 0.3, SE 1.2). Main [117], [118].
- LEAR here is refitted once a year on the last two years; the published LEAR refits daily (card [1]). Main [107].
- Lasso-then-XGBoost: a Lasso fits the price on the linear columns; XGBoost fits the Lasso's errors on all columns; the prediction is the sum. Main [108]. Our own combination of cards [2] and [3].

### 7.2 Feature groups on Lasso-then-XGBoost (Main [121], [124], [125])

| Set | Saving share | MAE |
|---|---|---|
| calendar | 84.0% | 38.4 |
| fundamentals | 69.4% | 30.4 |
| calendar + fundamentals | 64.4% | 39.7 |
| fundamentals + price memory | 95.3% | 18.3 |
| calendar + price memory | 85.5% | 26.2 |
| calendar + fundamentals + price memory | 95.8% | 19.1 |
| + level interaction | 96.2% | 19.1 |
| + neighbour spreads | 95.6% | 17.5 |

Decisions: price memory kept (gap 31.4 points, SE 4.1); fundamentals kept (10.3, SE 2.4); calendar ties (0.5, SE 0.7) and stays only because this comparison ran after 2025 was scored; level interaction ties (0.4, SE 0.8), dropped as the simpler; neighbour spreads lose in both years (-0.6, SE 0.3), dropped though they cut MAE from 19.1 to 17.5. Main [124], [125].

### 7.3 Tuning (Main [128] to [165])

- First grid (Lasso penalty 0.3, 1, 3 x depth 3, 6; 200 trees at 0.1): all 95.4% to 96.6%; rule 2 picks penalty 1, depth 6 (96.5%, MAE 17.7). Main [128], [129].
- XGBoost alone, six settings: best depth 6, rate 0.05 (95.9%, MAE 18.3). Lasso-then-XGBoost leads it in both years (0.6 points, SE 0.4). Main [132] to [138].
- Wider grid, 24 tree settings each (x 3 penalties for the two-stage model, 72): Lasso-then-XGBoost best by rule 2 is penalty 3, depth 6, rate 0.03, 500 trees (96.7%, by year 96.3%, 97.1%, MAE 17.3); XGBoost alone's is depth 2, rate 0.03, 500 trees (95.7%, by year 95.5%, 96.0%, MAE 17.6). Gap 1.0 points, SE 0.1: Lasso-then-XGBoost ahead in both years, final. Main [144] to [147].
- Random search, 100 settings each: neither search's best replaces the wider-grid best under rule 3 (XGBoost search best 95.5%, MAE 17.0, ties; Lasso-then-XGBoost search best 96.7%, MAE 18.0, ties). Main [150] to [165].
- **Final model:** Lasso-then-XGBoost, Lasso penalty 3, 500 trees, depth 6, learning rate 0.03, 17 columns. CV saving share 96.7%, MAE 17.3, RMSE 27.9, MASE 0.628, right shift on 74.7% of days. Main [167], [185].

### 7.4 The rule change since the presentation (`docs/selection-rule-change.md`)

The presentation chose XGBoost alone (depth 3, learning rate 0.05, 200 trees) under an earlier rule that compared a gap in mean saving share with each model's fold SD. That SD mostly measured how different 2023 and 2024 are, which moves both models the same way. The report uses paired fold gaps, under which Lasso-then-XGBoost wins. Old against new: validation 95.6% vs 96.7%; 2025 test 96.4% vs 95.9%; 2025 lead over the rule 105,000 vs 83,000 EUR a year; 2026 reserve 98.7% vs 99.1%, reserve lead 49,000 vs 73,000; reserve interval -11,000 to 141,000 vs 20,000 to 157,000. Changed on 2026-10-30, after 2025 had been scored.

## 8. Diagnostics (final model, out-of-fold 2023 and 2024; Main [174] to [188])

| Fact | Value | Source |
|---|---|---|
| Error grows with level | mostly within ±50 EUR/MWh below a prediction of 150; about -340 to +430 above 300 | Main [174] |
| By hour | mean residual -3 to +2 EUR/MWh; evening peak (19:00 to 20:00) and nights slightly too low; 09:00 to 16:00 about 3 too high | Main [177] |
| Daily errors | within ±40 EUR/MWh on 96% of days; worst: +110 and +180 on 11 and 12 Dec 2024, -245 the day after (windless cold spell), -140 on 7 Nov 2024 | Main [180] |
| By price band | ordinary hours (0 to 200) MAE 16.1, 16,545 hours; negative hours 19.4, 758 hours; above 200: 91.1, 241 hours, 56.3 too low on average | Main [182] |
| asinh target | tried, worse: 94.5% vs 96.7%, behind in both years (gap 2.2, SE 0.7), MAE 18.2 vs 17.3 | Main [185], [187], [188] |
| CV lead over the rule | 2.7 points (96.7% vs 94.1%), SE 0.7, ahead in both years | Main [187] |

## 9. Test: 2025 (Main [191], [195])

| Method | Saving share | EUR a year | Shift cost EUR/MWh | Right shift | MAE | RMSE | MASE |
|---|---|---|---|---|---|---|---|
| Final model | 95.9% | 4,287,000 | 59.84 | 73.7% | 16.2 | 24.1 | 0.625 |
| Residual-load rule | 94.1% | 4,205,000 | 60.30 | 70.7% | | | |
| Same hour yesterday | 83.2% | 3,717,000 | 62.97 | 49.9% | 26.0 | 40.4 | 1.000 |
| Last year's profile | 68.3% | 3,055,000 | 66.60 | 32.9% | | | |
| Standard shift | 0% | 0 | 83.34 | 1.4% | | | |
| Training mean | -33.6% | -1,503,000 | 91.57 | 6.6% | 35.8 | 52.6 | 1.377 |
| Hindsight | 100% | 4,469,000 | | | | | |

- Lead over the rule: 82,645 EUR a year, "83,000". 1.8 points (CV: 2.7). Main [192], [195].
- Hindsight saving 4,469,477 EUR (4.47 million). Week [32] (rest table) and Main [208].
- Choosing the shift at all, by the rule, is worth 4.2 million a year, about 50 times the model's extra 83,000. Main [197].
- Block bootstrap of the daily lead, 10,000 resamples (card [5]): 1-day runs 2,700 to 170,000, 7-day 3,700 to 181,000, 30-day 550 to 198,000 EUR a year; model no better than the rule in 2.1%, 2.0%, 2.4% of resamples. Main [195], [196].
- Running lead: model trails the rule through February, ahead from March, near 40,000 through summer, about 80,000 in September, ends at 83,000. Main [208]; fig3.
- Typical day 6 March 2025 (weekday closest to 2025's median daily mean): model's shift 08:00 to 16:00 = cheapest shift; forecast 7 to 69 EUR/MWh too high in every hour. Main [212]; fig4.

## 10. Reserve: 2026-01-01 to 2026-10-11 (Main [200], [203])

Final model refitted on 2019-01-31 to 2025-12-31, scored once: 99.1% vs rule 97.8%, lead 1.3 points, 72,992 EUR a year ("73,000"); MAE 23.3, MASE 0.740; same hour yesterday 89.8%. Bootstrap (7-day runs) 20,000 to 157,000; model no better than the rule in under 0.05% of resamples. 284 days.

## 11. Week-ahead schedule (`week_ahead.ipynb`, Dr Lim's first question)

| Fact | Value | Source |
|---|---|---|
| Set-up | all seven shifts fixed on the Sunday before at 11:55; final model and settings unchanged; inputs: calendar, price at the same hour on the decision Sunday, 7 and 14 days back, decision Sunday's mean/min/max, 7- and 30-day means up to it; no TSO forecasts: no TSO publishes wind or solar forecasts a week ahead; a week-ahead load forecast (daily max and min, published Fridays, Reg. 543/2013 Art. 6(1)(c)) exists on ENTSO-E but not in Energy-Charts. **Correction 2026-11-14:** the task brief and the first Claude draft said no week-ahead TSO forecast exists; only wind and solar have none | Week [1], [8] |
| Leak check | no week-ahead column reads past the decision Sunday | Week [14] |
| Validation | week-ahead 86.1% (3.50 million) vs day-ahead 96.7% (3.94 million); by year 82.8% (2023), 89.4% (2024) | Week [28], [29] |
| 2025 | week-ahead 90.7%, 4,055,000 EUR a year; day-ahead 95.9%, 4,287,000; last year's profile 68.3%, 3,055,000 | Week [28] |
| Cost of a week's notice | about 233,000 EUR a year on 2025 (230,000 in the notebook text), about 438,000 on validation | Week [28]; computed from the table |
| Gap closed vs last-year profile | 81% on 2025, 74% on validation | Week [40] |
| Week-ahead MAE / MASE 2025 | 26.6 EUR/MWh, 1.03 (worse than same hour yesterday, yet picks well) | Week [28] |

## 12. Rest between shifts (Arbeitszeitgesetz §5, card [7]; Main 8.4, Week [32]; `docs/rest-check.md`)

Rule: 11 hours off between shifts, so with 8-hour shifts a start may be at most 5 hours earlier than the day before's. With rest kept, each forecast schedule picks day by day (as at 11:55 D-1); hindsight with rest is the cheapest rest-keeping sequence over the year (dynamic programming) and is the denominator.

| Schedule, 2025 | Rest breaks | Share of days | Breaks, validation | Saving share | Saving share, rest kept | Change, EUR a year |
|---|---|---|---|---|---|---|
| Final model (day-ahead) | 27 | 7.4% | 71 (9.7%) | 95.9% | 91.7% | -279,000 |
| Week-ahead | 12 | 3.3% | 40 (5.5%) | 90.7% | 89.8% | -129,000 |
| Residual-load rule | 43 | 11.8% | 106 (14.5%) | 94.1% | 85.8% | -453,000 |
| Hindsight | 26 | 7.1% | 72 (9.8%) | 100% | 100% | -97,000 |

With rest kept: final model saves 4.01 million (4,008,643); leads the rule by about 257,000 a year (instead of 83,000) and the week-ahead schedule by about 83,000 (instead of 233,000). Clock-change days count clock hours, so real rest is one hour shorter (March) or longer (October).

## 13. Plan the week, adjust the day before (`week_ahead.ipynb` section 6, Dr Lim's next point; `docs/week-flex.md`)

Set-up: planned start = week-ahead model's choice (fixed on the Sunday before). Final start = final model's cheapest forecast shift among starts within W hours of the plan (either way), or 0 to W hours earlier only, W = 0 to 4. With rest kept, the plan is the rest-kept week-ahead schedule and the final start also keeps 11 hours after the previous day's final shift; a legal start always exists. Week [34], [36].

| Reach | Free, 2025 | EUR a year | Rest kept, 2025 | EUR a year | Free, validation | Rest kept, validation | Days moved 2025 (free) | Mean move h |
|---|---|---|---|---|---|---|---|---|
| 0 h (week-ahead plan) | 90.7% | 4,055,000 | 89.8% | 3,926,000 | 86.1% | 89.9% | 0% | 0 |
| 1 h either way | 93.1% | 4,160,000 | 92.5% | 4,045,000 | 87.9% | 92.2% | 35.1% | 0.35 |
| 2 h either way | 93.3% | 4,172,000 | 92.8% | 4,057,000 | 88.1% | 92.5% | 35.1% | 0.40 |
| 3 h either way | 93.5% | 4,177,000 | 92.8% | 4,060,000 | 88.5% | 93.0% | 35.9% | 0.47 |
| 4 h either way | 93.8% | 4,191,000 | 93.3% | 4,078,000 | 88.9% | 93.3% | 36.4% | 0.54 |
| 1 h earlier only | 91.8% | 4,104,000 | 91.1% | 3,984,000 | 86.7% | 91.0% | 12.9% | 0.13 |
| 4 h earlier only | 92.3% | 4,125,000 | 91.8% | 4,012,000 | 87.3% | 92.4% | 13.2% | 0.20 |
| Any start (day-ahead choice) | 95.9% | 4,287,000 | 91.7% | 4,009,000 | 96.7% | 85.3% | | |

Validation EUR a year: free 0 h 3.50 million, 1 h 3.58, 4 h 3.62, day-ahead 3.94; rest kept 0 h 3.47, 1 h 3.56, 4 h 3.61, day-ahead 3.29. Week [36].

Readings: free, 1 h either way wins back 105,000 of the 233,000 EUR a year a week's notice costs on 2025 (45%); 4 h wins back 136,000 (59%). On validation 74,000 and 118,000 of 438,000 (17% and 27%). Earlier only wins back roughly half as much (49,000 at 1 h, 70,000 at 4 h on 2025). With rest kept, every either-way reach of 1 h or more beats choosing each day the day before: 4.04 million (1 h) and 4.08 million (4 h) vs 4.01 million on 2025 (+36,000 and +69,000); 3.56 and 3.61 vs 3.29 million on validation. Likely reason: day-by-day choice under the rule can take a late shift that rules out a cheap early one tomorrow; a plan that already keeps the rule keeps each day's choice near a workable sequence. Main 8.4 makes the same point for the greedy day-ahead schedule. Fig5.

## 14. Things the report must state

1. TSO wind and solar forecasts are published after the gate (18:00 D-1); results assume a vendor forecast of similar quality bought the same morning. The residual-load rule uses the same forecasts.
2. The selection rule changed after the presentation and after 2025 was scored (section 7.4); the calendar decision too ran after 2025 was scored.
3. Only the final model is scored on 2025 and 2026; LEAR and other candidates never are.
4. Simplifications: shift within one day; premiums do not stack; weekend premium on every weekend hour (raises cost of weekend shifts, never changes which wins); price taker.
5. From 2025-10-01 the target is the hourly mean of four quarter-hour prices.
6. No fuel price and no public holidays in the data.
