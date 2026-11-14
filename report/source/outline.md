# Outline: DBA3803 group report

Reader: Dr Lim Ek Hong, a marker who knows predictive analytics but has not seen the code; he saw the presentation and asked two follow-ups (a week-ahead schedule, and rest between shifts) and then suggested planning a week ahead with a small daily adjustment. He reads for the business problem, sound modelling choices, results in business terms, a critical look at the model, and recommendations (brief: `project-info.pdf`). No page limit is given; aim for about 12 to 15 pages with five figures and four tables. Structure follows the Citi Bike report (`skills/report-writer/references/structure.md`): business problem with an "in short" paragraph, data and what is missing, exploration that justifies the method, method, selection, test, practice, recommendations, limitations. Every number from `facts.md`; every citation from `sources.txt`.

## 1. The business problem

Point: a German plant that runs one 8-hour shift a day can save millions a year by choosing when the shift runs, and a price forecast decides that choice better than the simple rule.
- Concrete set-up: 500 MWh a day, 62.5 MW an hour; at 11:55 on D-1, five minutes before the day-ahead auction closes at 12:00 [9], it picks a start from 00:00 to 16:00. Cost = day-ahead price plus a staff premium of 625 EUR an hour (10 EUR/MWh) at night and weekends.
- What is done today: a fixed shift; we use 06:00 to 14:00 as the standard. The cheapest shift in hindsight saves 4.47 million EUR a year against it in 2025.
- The simple alternative without a price model: pick the hours with the lowest forecast residual load. It cannot weigh the premium in euros.
- Questions, numbered: (1) Can a price forecast made at 11:55 pick a cheaper shift than the residual-load rule? (2) How much is that worth a year? (3) Does it still pay if shifts are fixed a week ahead, and if staff must get 11 hours of rest [7]?
- "In short" paragraph with the headline numbers: 95.9% of the possible saving on 2025, 4.29 million EUR a year, 83,000 more than the rule; a week's notice costs about 233,000; with rest kept, planning on Sunday and letting the 11:55 forecast move the start by up to an hour saves 4.04 million, more than choosing each day freely under the rule (4.01 million).
No figure.

## 2. Data, assumptions and what is missing

Opener: everything rests on public data, so first what it covers and what it leaves out.
- Table 1: sources (Energy-Charts [10]: DE-LU prices, TSO day-ahead forecasts of load, solar, wind, 11 neighbouring zones), span, rows.
- Cleaning, counted: four steps (average to UTC hours since prices turn quarter-hourly on 1 Oct 2025 [9]; calendar in German time; forward-fill the one missing Dutch day; keep every extreme, including the -500 hour).
- Split: two validation folds (2023, 2024), test 2025, reserve 2026 to 11 Oct; why chronological; exploration used 2019 to 2023.
- Gaps and assumptions, ranked: (1) forecast timing [8]: load is published before the gate, wind and solar after it; a vendor forecast stands in; (2) quarter-hour target from Oct 2025; (3) no fuel price, no holidays; (4) price taker with the 0.18 EUR/MWh number; (5) shift within one day, premiums do not stack.
No figure.

## 3. What decides the cheapest shift

Opener: before modelling, what moves the cheapest shift and how much simple rules already get.
- Figure 1 (fig1-cheapest-start.png): cheapest start is 00:00 or 09:00 to 10:00; 06:00 on 1% of days. So the standard shift is almost never right, and the choice is worth money.
- Shape not level: only the within-day pattern changes the choice; in 2021 and 2022 over 80% of price movement was whole days moving, so a model judged on MAE spends its effort on the wrong part. This sets the metric (section 4).
- Residual load sets the shape (within-day correlation 0.70 pooled), but its price slope moved from 1.1 to 7.1 EUR/MWh per GW, and no fuel price is in the data, so recent prices must carry the level.
- Simple rules on 2019 to 2023: residual-load rule 87% to 97%, yesterday's prices 52% to 71%, last year's profile 26% to 54%. The rule misses most in calm years because it ignores the premium. That gap is the room for a forecast.
Bridge: so the method forecasts prices with residual load and recent prices, and is judged by the shift it picks.

## 4. Method

Opener: section 3 showed the decision uses only the shape, so the method is judged by the shift it picks, not by its price error [6].
- 4.1 The decision metric: saving share formula; EUR a year; MAE, RMSE, MASE [4] reported but deciding only ties. Worked example with 2023's numbers (74.5, 97.4, 76.2).
- 4.2 Three candidate methods: Lasso-then-XGBoost (Lasso [3] for the steady pattern, XGBoost [2] on its errors), XGBoost alone, LEAR [1] as the field's benchmark (refitted yearly here, daily in the original). Why the simple approach (a linear model alone) fails: the residual-load slope bends with the price level.
- 4.3 How options are chosen: paired gaps per validation year, SE, the three rules. Then the rule change since the presentation, in two or three sentences, with old and new results.
No figure.

## 5. Choosing the model on 2023 and 2024

Opener: one sentence on what was compared and how a choice was made; then one subsection per decision.
- 5.1 Methods: Table 2 (methods at default settings). LEAR has the lowest MAE but the lowest saving share of the three; the residual-load rule beats it too. Ends: LEAR dropped.
- 5.2 Feature groups: price memory and fundamentals kept; calendar ties and stays (run after 2025 was scored); level interaction tied and dropped; neighbour spreads lower saving share though they cut MAE. Ends: 17 columns.
- 5.3 Tuning: wider grid; Lasso-then-XGBoost 96.7% vs XGBoost alone 95.7%, ahead in both years; random searches change nothing. Ends: final model and settings.
- 5.4 Is it sensible: diagnostics: spikes above 200 missed by 91 EUR/MWh; evening peak a bit low; asinh tried and worse. Ends: known weakness, kept.
No figure (tables carry it).

## 6. Results on 2025

Opener: with everything fixed on the validation years, the final model is refitted on 2019 to 2024 and scored once on 2025.
- Figure 2 (fig2-saving-share-2025.png) and Table 3: 95.9% vs 94.1%, 4.29 million, 83,000 more than the rule; CV lead held.
- Is the lead luck: block bootstrap [5], interval 3,700 to 181,000 (7-day runs); model no better than the rule in 2% of resamples.
- Figure 3 (fig3-running-lead-2025.png): lead builds over many days.
- Figure 4 (fig4-one-day.png): 6 March 2025, forecast too high in every hour, still the cheapest shift: the shape is what counts.
- Reserve 2026: 99.1% vs 97.8%, 73,000 a year, interval 20,000 to 157,000.

## 7. Putting it into practice: notice and rest

Opener: Dr Lim's two questions after the presentation, and his follow-up suggestion.
- 7.1 A week's notice: 90.7%, 4.05 million, about 233,000 less than day-ahead on 2025, a million more than last year's profile. No week-ahead TSO forecast exists, so the plan has no weather.
- 7.2 Rest between shifts: Table 4 (rest check). Free choice breaks the rule on 7.4% of days; keeping it costs the model 279,000, the rule 453,000.
- 7.3 Plan the week, adjust the day before: Figure 5 (fig5-plan-then-adjust.png) and Table 5. One hour either way wins back about half of a week's notice when choosing freely; with rest kept it beats choosing each day freely. Earlier-only gives roughly half the gain.

## 8. Recommendations (ranked by impact)

1. Choose the shift every day instead of fixing it at 06:00 (rule alone: 4.20 million a year on 2025).
2. Plan each week on Sunday with the week-ahead model and let the 11:55 forecast move each start by up to one hour either way, keeping 11 hours of rest (4.04 million, with a week's notice and the law kept).
3. Use the price forecast rather than the residual-load rule for the daily choice (+83,000 free choice, +257,000 with rest kept).
4. Buy a vendor wind and solar forecast available before 12:00.

## 9. Limitations (counted, each with what a follow-up needs)

1. Forecast timing (wind, solar after the gate): needs a vendor forecast to test.
2. Small lead, two validation years: more years or a rolling evaluation.
3. Spikes missed: a spike model or scarcity features.
4. Simplified decision: shifts crossing midnight, stacked premiums.
5. No fuel price or holidays: add gas prices and the `holidays` calendar.
6. Selection rule and calendar decision made after 2025 was scored; the 2026 reserve is the clean check.
7. LEAR refitted yearly, likely understated.

## References

The ten cards in `sources.txt`, author-date style.
