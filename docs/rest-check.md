# Rest between shifts: text for the report

**Out of date since 2026-11-15.** The model-based numbers here come from Lasso-then-XGBoost, the final model under the paired-gap rule, which was reverted. With XGBoost alone as the final model, the current numbers are in the notebooks it names and in `docs/group-report-numbers.md`. Rule-only numbers (the `last-week` rule on its own) did not change.

Source: `de_price_forecast.ipynb` section 8.4 and `week_ahead.ipynb` section 5, 2025 test year.

We checked every schedule against the German rule of 11 hours of rest between shifts (Arbeitszeitgesetz §5), which an 8-hour shift breaks when it starts more than 5 hours earlier than the day before's. Choosing each day freely, the final model breaks it on 27 days of 2025 (7.4%), the residual-load rule on 43, the week-ahead schedule on 12, and even perfect hindsight on 26. When each day instead takes its cheapest forecast shift among the starts that keep 11 hours of rest, the final model saves about 279,000 EUR a year less: 4.01 million against the standard shift, 91.7% of what the best rest-keeping schedule in hindsight saves. The rule costs the residual-load rule more (453,000 EUR a year) and the week-ahead schedule less (129,000), so with rest kept the model leads the rule by about 257,000 EUR a year and the week-ahead schedule by about 83,000.

| Schedule | Rest breaks, 2025 | Share of days | Saving share | Saving share with rest | Change in saving, EUR a year |
|---|---|---|---|---|---|
| Final model (day-ahead) | 27 | 7.4% | 95.9% | 91.7% | -279,000 |
| Week-ahead | 12 | 3.3% | 90.7% | 89.8% | -129,000 |
| Residual-load rule | 43 | 11.8% | 94.1% | 85.8% | -453,000 |
| Hindsight | 26 | 7.1% | 100% | 100% | -97,000 |

With rest kept, each forecast schedule picks day by day, which matches a choice made at 11:55 the day before. Hindsight with rest kept is the cheapest rest-keeping sequence over the whole year, found by dynamic programming, and is the denominator of the saving share with rest.
