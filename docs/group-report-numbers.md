# The group report's numbers against the repo

Checked on 2026-11-15 against `report/group/DBA3803_Group10_Report_v3.docx`, after the repo went back to XGBoost alone as the final model (depth 3, learning rate 0.05, 200 trees) under the fold-SD rule, and both notebooks were rerun.

**2026-11-16, v4.** `report/group/DBA3803_Group10_Report_v4.docx` takes the repo value in every row that was a mismatch or not in repo (Martin's call: the repo's numbers everywhere), rounded as the report rounds. Permutation importance is now computed in the main notebook (section 6, `permutation_importance` in `lib/evaluation.py`: the final model fitted per fold, each column shuffled in the validation year, five shuffles, mean of 2023 and 2024), and the rule's MAE for Table E1 in `wa` 8. Figures 5 and E1 are redrawn from the repo (`report/group/source/build_v4_figures.py`); the edits are scripted in `report/group/source/make_v4.py`. The Status column below is against v4; the Report column still quotes v3. v4 also corrects the forecast timing: only the wind and solar forecasts come at 18:00, the load forecast is out before the gate.

Sources: `main` is `de_price_forecast.ipynb`, `wa` is `week_ahead.ipynb`, each by section. The report's section 8 and appendix E divide every rest-keeping schedule by the free daily hindsight; the repo values in those rows use the same denominator (`saving_share` in `wa` 8), so they can replace the report's one for one. `wa` 5 to 7 use the rest-keeping hindsight instead (`saving_share_rest`).

Status: **match** means equal at the report's rounding. **mismatch** carries its cause. **not in repo** means the repo has no cell for it.

## Causes of the mismatches

- **A. Row-shifted lag.** The teammate's section 9 script picks the weekly rule's start on `price_lag168`, shifted by 168 rows, which reads one hour off around the clock changes; the repo looks each price up by German date and hour (`docs/week-rule.md`). Moves every row built on last week's prices by a few tenths of a point and up to 20,000 EUR a year.
- **B. Different week-ahead model.** The teammate's week-ahead XGBoost has its own feature set; the repo's (`wa` 2) uses the decision Sunday's prices, lags of 7 and 14 days and the Sunday's statistics, with the final model's settings. Same idea, different forecasts, so different weekly starts.
- **C. Not computed in the repo.** Permutation importance is in neither the repo nor `c88eb46`. A scratch check (final model, one shuffle per column, seed 0, mean of the two validation years, not in a notebook) gives the same order at the top and similar sizes; the exact values move with the shuffle.
- **D. Follows from A.** A bootstrap or gap whose one side is the weekly rule's row moves with that row.

## Section 4.2

| Report | Repo | Source | Status |
|---|---|---|---|
| Residual load within-day correlation 0.70, at most 0.29 for any single forecast | 0.70, 0.29 | main 3.4 | match |
| Slope 1.1 EUR/MWh per GW in 2019, 7.1 in 2022 | 1.1, 7.1 | main 3.4 | match |
| Same hour yesterday correlation 0.90, weekly echo at 168 h | 0.90, peak at 168 | main 3.5 | match |
| Sundays 27 EUR/MWh below their week | 27 | main 3.8 | match |
| Neighbours 0.906 against Germany's lag 0.903 | 0.906, 0.903 | main 3.6 | match |
| Residual-load rule 87–97% of the possible saving | 87% to 97% | main 3.10 | match |
| Final model uses 17 columns | 17 | main 5.3, column table | match |
| Calendar plus forecasts 64.4%; adding price memory 95.8% | 0.644, 0.958 | main 5.2 | match |
| Level interaction +0.4 points, neighbour spreads −0.6 points | 0.962 − 0.958, 0.956 − 0.962 | main 5.2 | match |

## Section 6

| Report | Repo | Source | Status |
|---|---|---|---|
| Table 4, Lasso-then-XGBoost: 95.6%, fold SD 1.3, MAE 17.5, RMSE 26.9, MASE 0.64 | 0.956, 0.013, 17.486, 26.945, 0.636 | main 5.1 | match |
| Table 4, XGBoost: 95.4%, 0.3, 18.9, 28.3, 0.69 | 0.954, 0.003, 18.854, 28.275, 0.686 | main 5.1 | match |
| Table 4, residual-load rule 94.1% | 0.941 | main 5.1 | match |
| Table 4, LEAR: 93.5%, 1.2, 17.1, 28.8, 0.62 | 0.935, 0.012, 17.094, 28.845, 0.621 | main 5.1 | match |
| Table 4, same hour yesterday: 73.7%, MAE 27.5, RMSE 42.3, MASE 1.00 | 0.737, 27.513, 42.301, 1.000 | main 5.1 | match |
| Table 4, last year's profile 55.3% | 0.553 | main 5.1 | match |
| After tuning 96.7% against 95.6%, inside the fold SD of 1.2; MAE 17.3 against 17.0 | 0.967 (SD 0.006) against 0.956 (SD 0.012); 17.282 against 16.961 | main 5.3, wider grid | match |
| Table 5, XGBoost: 96.4%, €4.31M, 71.5% of days, MAE 16.4 (MASE 0.63) | 0.964, 4,309,336, 0.715, 16.404 (0.632) | main 7.1 | match |
| Table 5, residual-load rule: 94.1%, €4.20M, 70.7% | 0.941, 4,204,680, 0.707 | main 7.1 | match |
| Table 5, same hour yesterday: 83.2%, €3.72M, 49.9%, 26.0 (1.00) | 0.832, 3,716,954, 0.499, 25.965 (1.000) | main 7.1 | match |
| Table 5, last year's profile: 68.3%, €3.05M, 32.9% | 0.683, 3,054,707, 0.329 | main 7.1 | match |
| Table 5, standard shift: 0%, €0, 1.4% | 0.000, 0, 0.014 | main 7.1 | match |
| Perfect hindsight saves €4.47M | 4,469,477 | main 8.1; wa 5, hindsight row | match |
| Standard shift costs about €15.2M a year, 83.3 EUR/MWh on 182,500 MWh | 15,208,612; 83.335; 500 × 365 | wa 8, standard-shift 2025; main 7.1 `shift_cost` | match |
| Cuts the cost by €4.31M, or 28% | 4,309,336 / 15,208,612 = 28.3% | wa 8 | match |
| Beats the rule by 2.3 points, about €105,000 a year | 0.964 − 0.941; 104,657 | main 7.1, 7.2 | match |
| 1.5-point lead in validation | 0.956 − 0.941 | main 5.1, 5.3 | match |
| 7-day bootstrap interval about €8,000 to €223,000; no better in under 3% of resamples | 8,136 to 223,262; 1.6% (2.7% at 30 days) | main 7.2 | match |
| 2026 reserve 98.7% against 97.8%, about €49,000 a year | 0.987, 0.978, 49,152 | main 7.4 | match |
| €4.2M even with the simple rule | 4,204,680 | main 7.1 | match |
| Ordinary and negative hours missed by about 16 EUR/MWh | 15.9, 16.5 | main 6, price bands | match |
| 241 hours above 200 EUR/MWh missed by 89, predicted 61 too low | 241, 89.1, +61.0 | main 6, price bands | match |
| Evening peak 19:00–20:00 predicted slightly too low | the same | main 6, residual by hour | match |
| Largest misses in a windless cold spell in December 2024 | the same | main 6, daily residual | match |
| asinh target 93.1% against 95.6% | 0.931, 0.956 | main 6, asinh test | match |
| Spikes about 1.4% of hours | 241 / 17,544 = 1.4% | main 6, price bands | match |
| Exact cheapest shift on 71.5% of days | 0.715 | main 7.1 | match |

## Section 7

| Report | Repo | Source | Status |
|---|---|---|---|
| Shuffling residual load costs 64 points and raises MAE by about 20 EUR/MWh | 61.0 points, +19.6 | main 6, permutation importance | v4: 61 points, about 20 |
| Yesterday's and last week's price at the same hour cost 3.1 and 1.9 points | 2.7 and 1.4 | main 6, permutation importance | v4: 2.7 and 1.4 |
| Then the 30-day mean and yesterday's maximum | yesterday's maximum 1.3, then the 30-day mean 0.7 | main 6, permutation importance | v4: order swapped to match |
| Price memory removed as a group costs 31 points | 0.958 − 0.644 = 31.4 | main 5.2 | match |

## Section 8

| Report | Repo | Source | Status |
|---|---|---|---|
| €4.31M, 96.4%, €105,000 more than the rule | as section 6 | main 7.1 | match |
| Daily plan changes the start on 135 days | 135 | wa 8, daily-xgboost 2025 | match |
| Breaks the rest rule on 20 days, 5 of them a 16:00 shift then a 00:00 start | 20, 5 | wa 8, rest-break count | match |
| Validation: last week's best hour 86.4% against week-ahead XGBoost 84.4%, fold SD 2.5 | 0.869 (SD 0.011) against 0.838 (SD 0.052); a tie under rule 1 either way | wa 8, validation table | v4: 86.9% against 83.8%, fold SD 5.2 |
| Table 6, daily XGBoost: 96.4%, €4.31M, 20 rest breaks, 135 start changes | 0.964, 4,309,336, 20, 135 | wa 8 | match |
| Table 6, daily XGBoost + rest rule: 91.0%, €4.07M, 0, 154 | 0.910, 4,069,121, 0, 154 | wa 8 | match |
| Table 6, weekly, last week's best hour: 89.9%, €4.02M, 0, 21 | 0.895, 4,001,426, 0, 19 | wa 8, weekly-last-week-rest | v4: 89.5%, €4.00M, 0, 19 |
| Table 6, weekly, week-ahead XGBoost: 90.1%, €4.02M, 0, 16 | 0.900, 4,021,970, 0, 11 | wa 8, weekly-week-ahead-model-rest | v4: 90.0%, €4.02M, 0, 11 |
| Table 6, weekly, perfect foresight: 91.8%, €4.10M, 0, 19 | 0.918, 4,104,060, 0, 19 | wa 8, weekly-hindsight-rest | match |
| Weekly gives up about €293,000 against the free daily plan, interval €135,000 to €458,000 | 307,910; 7-day 149,644 to 483,064 | wa 8, bootstrap | v4: €308,000, €150,000 to €483,000 |
| Daily plan with rest leads the weekly by €53,000, interval −€151,000 to €216,000 | 67,695; 7-day −137,494 to 244,199 | wa 8, bootstrap | v4: €68,000, −€137,000 to €244,000 |
| 2026 reserve, weekly: 92.9%, €5.32M | 0.931, 5,329,625 | wa 8, weekly-last-week-rest 2026 | v4: 93.1%, €5.33M |
| Table 7, today: €15.21M | 15,208,612 | wa 8, standard-shift 2025 | match |
| Table 7, weekly last week's best hour: €11.19M, saving €4.02M, cut 26.4% | 11,207,186; 4,001,426; 26.3% | wa 8 | v4: €11.21M, €4.00M, 26.3% |
| Table 7, daily XGBoost with rest rule: €11.14M, €4.07M, 26.8%, 154 start changes | 11,139,491; 4,069,121; 26.8%; 154 | wa 8 | match |
| Table 7, daily XGBoost unrestricted: €10.90M, €4.31M, 28.3%, 20 days of short rest | 10,899,276; 4,309,336; 28.3%; 20 | wa 8 | match |
| The weekly plan costs about 7% of the daily model's saving | (4.309 − 4.001) / 4.309 = 7.1% | wa 8 | match |
| 62.5 MW is about 0.1% of the 54 GW average demand | mean `fc_load` 54,056 MW, all 15-minute rows of the raw file; 62.5 / 54,056 = 0.12% | scratch, `data/de_lu_day_ahead_forecasts.parquet` | match (not in a notebook cell) |

## Appendix C

| Report | Repo | Source | Status |
|---|---|---|---|
| C1: the 17 columns and their groups | the same 17 | main 5.3, column table | match |
| C2, calendar only: 84.0%, fold SD 0.9, MAE 38.4 | 0.840, 0.009, 38.354 | main 5.2 | match |
| C2, fundamentals only: 69.4%, 6.6, 30.4 | 0.694, 0.066, 30.433 | main 5.2 | match |
| C2, calendar + fundamentals: 64.4%, 5.7, 39.7 | 0.644, 0.057, 39.717 | main 5.2 | match |
| C2, calendar + price memory: 85.5%, 3.2, 26.2 | 0.855, 0.032, 26.151 | main 5.2 | match |
| C2, fundamentals + price memory: 95.3%, 0.8, 18.3 | 0.953, 0.008, 18.316 | main 5.2 | match |
| C2, all three: 95.8%, 0.2, 19.1 | 0.958, 0.002, 19.128 | main 5.2 | match |
| C2, + level interaction: 96.2%, 19.1 | 0.962, 19.094 | main 5.2 | match |
| C2, + neighbour spreads: 95.6%, 17.5 | 0.956, 17.486 | main 5.2 | match |
| C3, XGBoost grid best: depth 3, rate 0.05, 200 trees, 95.6%, MAE 17.0 | the same, 0.956, 16.961 | main 5.3 | match |
| C3, Lasso-then-XGBoost wider grid: penalty 3, depth 6, rate 0.03, 500 trees, 96.7%, 17.3 | the same, 0.967, 17.282 | main 5.3 | match |
| C3, XGBoost random search: depth 5, rate 0.03, 175 trees, L2 = 61, 95.5%, 17.0 | the same (L2 60.6), 0.955, 17.008 | main 5.3 | match |
| C3, Lasso-then-XGBoost random search: penalty 0.70, depth 5, rate 0.11, 306 trees, 96.4%, 17.6 | the same, 0.964, 17.637 | main 5.3 | match |
| C3, final XGBoost: 95.6%, 17.0 | 0.956, 16.961 | main 6, final model | match |

## Appendix D

| Report | Repo | Source | Status |
|---|---|---|---|
| D1, below 0: 758 hours, MAE 16.5, mean residual −5.0 | 758, 16.459, −5.047 | main 6, price bands | match |
| D1, 0 to 200: 16,545, 15.9, −0.6 | 16,545, 15.932, −0.607 | main 6, price bands | match |
| D1, above 200: 241, 89.1, +61.0 | 241, 89.103, 60.984 | main 6, price bands | match |
| D1, asinh lowers the saving share from 95.6% to 93.1% | 0.956, 0.931 | main 6, asinh test | match |
| D2, 1 day: €104,657, 8,005 to 212,778, 1.8% | 104,657, 8,005 to 212,778, 0.018 | main 7.2 | match |
| D2, 7 days: 8,136 to 223,262, 1.6% | 8,136 to 223,262, 0.016 | main 7.2 | match |
| D2, 30 days: −1,507 to 239,027, 2.7% | −1,507 to 239,027, 0.027 | main 7.2 | match |
| D2, 2026 reserve: €49,152, −10,562 to 140,605, 6.2% | 49,152, −10,562 to 140,605, 0.062 | main 7.4 | match |

## Appendix E

| Report | Repo | Source | Status |
|---|---|---|---|
| E1, last week's best hour: 86.4%, fold SD 2.5, MAE 33.2, MAE fold SD 0.6 | 0.869, 0.011; MAE 30.8, SD 0.3 (the rule's 7-day profile as a forecast) | wa 8, validation table; wa 8, E1 MAE cell | v4: 86.9%, 1.1, 30.8, 0.3 |
| E1, week-ahead XGBoost: 84.4%, 0.7, MAE 30.9, 5.1 | 0.838, 0.052; MAE 29.5, SD 4.7 | wa 8, validation table; wa 8, E1 MAE cell | v4: 83.8%, 5.2, 29.5, 4.7 |
| E1, day-ahead forecasts (reference): 87.9%, 3.1 | 0.880, 0.030 | wa 8, weekly-day-ahead-model-rest | v4: 88.0%, 3.0 |
| E1, perfect foresight (reference): 89.4%, 1.8 | 0.894, 0.017 (89.42%, 1.72) | wa 8, weekly-hindsight-rest | v4: 89.4%, 1.7 |
| E2, daily XGBoost: 98.7%, €5.65M, 4 rest breaks, 96 start changes | 0.987, 5,649,573, 4, 96 | wa 8, daily-xgboost 2026 | match |
| E2, daily XGBoost + rest rule: 98.0%, €5.61M, 0, 98 | 0.980, 5,614,454, 0, 98 | wa 8 | match |
| E2, weekly, last week's best hour: 92.9%, €5.32M, 0, 17 | 0.931, 5,329,625, 0, 15 | wa 8 | v4: 93.1%, €5.33M, 0, 15 |
| E2, weekly, week-ahead XGBoost: 93.0%, €5.33M, 0, 15 | 0.930, 5,328,181, 0, 10 | wa 8 | v4: 93.0%, €5.33M, 0, 10 |
| E2, weekly, perfect foresight: 96.9%, €5.55M, 0, 16 | 0.969, 5,549,116, 0, 16 | wa 8 | match |

## The missing row: the rule's week, moved up to an hour by the day-ahead model, rest kept

`last-week-plan-move-1h-rest` in `wa` 8: the `last-week` rule fixes one start a week on Sunday (rest kept at the week boundary), then at 11:55 the day before the final XGBoost moves that day's start up to an hour either way, keeping 11 hours after the previous shift. Saving share on the report's denominator (free daily hindsight); the rest-keeping denominator in brackets.

| Period | Saving share | Saved a year | Rest breaks | Start changes | Yearly cost |
|---|---|---|---|---|---|
| 2023 | 87.9% (93.3%) | €3.67M | 0 | 147 | €14.11M |
| 2024 | 90.9% (95.2%) | €3.61M | 0 | 126 | €10.28M |
| Validation mean | 89.4%, fold SD 2.1 (94.2%) | €3.64M | 0 | 136.5 | |
| 2025 | 91.9% (93.9%) | €4.11M (4,105,784) | 0 | 104 | €11.10M (11,102,828), a cut of 27.0% |
| 2026 reserve | 95.8% (96.4%) | €5.49M (5,486,343) | 0 | 94 | €11.50M |

On 2025 it beats the daily XGBoost with the rest rule (91.0%, €4.07M, 154 start changes) by €36,662 a year. The 7-day block bootstrap puts the 95% interval for that lead at −€127,495 to €231,769, and the schedule does no better in 31.5% of resamples (1-day runs: −103,166 to 185,915; 30-day runs: −181,791 to 121,551), so the lead cannot be told apart from luck. It matches the weekly perfect-foresight schedule (€4.10M) while still giving staff the week's plan on Sunday. On the 2026 reserve it falls behind the daily schedule with rest kept (95.8% against 98.0%) but beats every weekly schedule except perfect foresight.

The calls behind it (`wa` 7.4, fold-SD rule, validation years): the one-hour move leads the rule's plain week by 2.6 points against a larger fold SD of 1.3 with rest kept (1.8 against 1.6 choosing freely), so it stays; against the week-ahead model's plan with the same move it ties (1.1 against 1.3), and the simpler rule's plan stays.
