# Fact-check 1: Gemini's version against facts.md and the code

Checked 2026-11-14: `dba3803-report-gemini.md` (Gemini's text, left untouched, built as `report/dba3803-report-gemini.docx`) against `facts.md`, `sources.txt`, both notebooks and `lib/decision.py`. Every number was matched by script (`check_numbers.py`: no unknown numbers except Gemini's own 92.6 and the rounding of 1,000,000) and every claim by reading. Fixes are applied in `dba3803-report-gemini-factfixed.md` / `report/dba3803-report-gemini-factfixed.docx`, changing only the wrong words. Ranked by how much each would mislead a reader.

## Wrong

1. **Week-ahead inputs and forecasts (7.1).** Gemini says the week-ahead model uses "price lags through Sunday morning" and that "European TSOs do not publish load or renewable forecasts seven days in advance". Sunday's whole day of prices is known at 11:55 on Sunday (its auction cleared on Saturday), and TSOs do publish a week-ahead load forecast, a daily maximum and minimum every Friday (Reg. 543/2013 Art. 6(1)(c)); only wind and solar have none, and the load one is just not in Energy-Charts. The second half came from Claude's draft and the task brief, not from Gemini: it is now fixed in the draft, `facts.md`, the source card and `week_ahead` section 6 (commit 9ca3707).
2. **Why plan-then-adjust beats the daily choice under the rest rule (7.3).** Gemini states as fact that the Sunday plan is "a globally coordinated sequence". It is not: the plan is chosen day by day too (greedy), and only hindsight is optimised over the year. The reason in the notebook is a likely explanation (a late shift today can rule out a cheap early one tomorrow), and the fixed copy says so.
3. **"Statistically significant lead" (4.3).** The paired-gap rule with two years has one degree of freedom; the notebook calls it a rough test. Calling it significance overstates every model choice in section 5.
4. **92.6% (4.1).** Gemini recomputed the worked example from rounded costs and got 92.6%, then called 93.0% "reported on Fold 1". From the unrounded costs it is 93.0% (Main [91]).
5. **"Unconstrained daily scheduling under the identical legal rest constraint" (In short, 7.3, recommendation 2).** Contradictory: the 4.01 million is the day-by-day choice with the rest rule kept. Reworded in four places.
6. **Savings called "electricity cost reductions" and "power cost savings" (In short, section 6, recommendation 1, section 1).** Every saving is on electricity plus the staff premium, not electricity alone.
7. **Dr Lim's point misquoted (7 intro).** Gemini has him saying day-ahead scheduling "frequently violates" the rest rule. He raised that consecutive shifts may leave too little rest; the 7.4% figure is ours.
8. **Target after October 2025 called "average hourly revenue" (2, item 2).** It is the hourly mean of four quarter-hour prices.
9. **Diurnal bias "2 to 3 EUR/MWh" too low at night and evening (5.4).** The mean residual runs from -3 to +2, so the underprediction is up to about 2; the 3 is the midday overprediction (Main [177]).
10. **Forward fill (2, step 3).** "Forward-filled using observations from 9 October" reads as a day copied over; each hour takes the last known price, 23:00 on 9 October (`pl.col(...).forward_fill()`, Main [96]). Claude's draft had the same slip ("filled with the day before"); fixed there too.

## Overstated or invented, small

11. "Decisive lead in March" (6): the lead was about 25,000 EUR in March and dipped later; now "pulled ahead".
12. "Definitive independent verification" for nine months of reserve (4.3): now "the cleaner check".
13. "Frequently pushing clearing values below zero" (1): 2.3% of hours in 2019 to 2023; now "sometimes".
14. "Nocturnal generation troughs" (3): the night start wins on low demand and low night prices, not a generation trough.
15. LEAR refitted yearly "to align with the computational cadence" (4.2): invented reason; the notebook only says it is refitted like the other two.
16. "Tracking global natural gas markets" (3): the notebook says the level depends mostly on the gas price; "global" is not supported.
17. "Dutch TTF" as the gas series to add (limitation 5): not in any source card; removed.
18. "Public TSO renewable forecasts are legally published after gate closure" (recommendation 4): the regulation sets a deadline of 18:00, not a publication time; reworded.

## Checked and right

All five tables, every number in the In short paragraph, the data and split sizes, the exploration figures (shape shares, slopes, correlations, rule shares), the selection gaps and SEs, the final settings, the diagnostics, the 2025 and 2026 results and bootstrap intervals, the rest table, the plan-then-adjust numbers (105,000 of 233,000; 136,000; 49,000 and 70,000; 74,000 and 118,000 of 438,000; 4.04 and 4.08 against 4.01 million; 3.56 and 3.61 against 3.29), the rule-change numbers, and all eleven citations, each used only for the claim on its card. Gemini kept all five figures with their captions and added no source.
