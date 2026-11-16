# A decision rule for the week ahead

**Out of date since 2026-11-15.** The model-based numbers here come from Lasso-then-XGBoost, the final model under the paired-gap rule, which was reverted. With XGBoost alone as the final model, the current numbers are in the notebooks it names and in `docs/group-report-numbers.md`. Rule-only numbers (the `last-week` rule on its own) did not change.

Source: `week_ahead.ipynb` section 7 (helpers `trailing_hourly_mean` in `lib/modeling.py`, `week_mean_premium` and `pick_weekly` in `lib/decision.py`), and the `last-7-days-mean` reference in sections 5.1 and 7.1 of `de_price_forecast.ipynb`.

## The rule

`last-week`: on the decision Sunday at 11:55, take each clock hour's mean settled price over the 7 days up to and including that Sunday, add the premium averaged over a week (10 EUR/MWh from 22:00 to 06:00, 10 x 2/7 by day), and run the cheapest 8-hour start from Monday to Sunday. One start a week.

## Final numbers

| | 2023 | 2024 | Validation mean | 2025 | EUR a year, 2025 |
|---|---|---|---|---|---|
| Rule `last-week` | 87.8% | 88.3% | 88.1% | 89.0% | 3,980,000 |
| Week-ahead model | 82.8% | 89.4% | 86.1% | 90.7% | 4,055,000 |
| Day-ahead model | 96.3% | 97.1% | 96.7% | 95.9% | 4,287,000 |

- Rule minus model: +5.0 and -1.1 points, mean +2.0, SE 3.1: a tie under rule 1, so the simpler rule sets the week. No rule variant beats `last-week` on the validation years (table in section 7.1); only `last-week` is scored on 2025.
- Start changes a year (2023/2024/2025): rule 10/19/20, week-ahead model 104/127/101, day-ahead model 159/169/155.
- Rest rule: the rule breaks it 3/2/2 times a year, always at a week boundary. With rest kept it gets 91.4%/91.8%/91.5% of the rest-keeping hindsight saving and 4.00 million EUR a year on 2025, the same as the day-ahead model under the rule (4.01 million).
- Day-before move of up to 1 hour either way on the rule's plan: validation +2.0 points (SE 0.9) free, +2.7 (SE 0.7) with rest kept, so it stays. 2025: 4.10 million free, 4.13 million with rest kept (94.4% of the rest-keeping hindsight saving). Rule plan against the model plan, both with the move: tie free (+2.2, SE 2.4), rule ahead in both years with rest kept (+2.2). Selection: the rule's plan plus the one-hour move.
- Day-ahead, the same idea (each hour's mean over the 7 days up to D-1, `last-7-days-mean`): see the main notebook's 5.1 and 7.1 tables; the final model leads it on both validation years and on 2025.

## Why the teammate's section 9 shows 89.9%

His `last-week` (in `~/Downloads/de_price_forecast.py`, section 9.3) is the same rule: one start a week, picked on `price_lag168` plus each day's premium, summed over the week. That equals the 7-day profile plus the week-averaged premium. Three things differ, all checked by running his `start_costs`, `plan_shifts` and `score_plan` on this repo's data:

1. **Rest kept.** His 89.9% has the rest rule on. With it off, his planner gets 89.4% on 2025.
2. **Denominator.** His saving share with rest kept still divides by the free daily hindsight. This repo divides by the best rest-keeping schedule, which gives 91.5% for the same schedule. On his denominator this repo's rest-kept rule gets 89.5% (4.00 million EUR a year).
3. **Row-shifted lag.** `price_lag168` shifts by 168 rows, which reads one hour off around the clock changes. Swapping in the profile looked up by German date and hour, his planner gives 4.00 million on 2025, exactly this repo's number; with his lag it gives 4.02 million. The rest-kept plan carries each week's start into the next, so one different week can move later weeks too.

His validation figure, 86.4%, is the same rule with rest kept, on his denominator, with the row-shifted lag (84.7% and 88.2%).

The 86.5% on 2025 in the scratch check of 2026-11-14 is a different rule: it added each day's own premium, so in 4 to 6 weeks a year the weekend started at another hour than the weekdays. That is `last-week-by-day` in section 7.1, which ties `last-week` on the validation years (88.1% each) and loses to it on simplicity. It is not scored on 2025 in the notebook.

His older pipeline (XGBoost alone as the daily model, 96.4%) does not touch the rule's numbers.
