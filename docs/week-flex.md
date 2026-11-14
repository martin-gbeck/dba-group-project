# Plan the week, adjust the day before: text for the report

Source: `week_ahead.ipynb` section 6, 2025 test year and the 2023 and 2024 validation years. Helpers `pick_near_plan` and `flex_check` in `lib/decision.py`.

Crews need their shifts well ahead, and the day-ahead forecast picks the better shift, so we combined the two. Each day's shift is planned on the Sunday before with the week-ahead model, and at 11:55 the day before, the final day-ahead model may move that day's start within a few hours of the plan: up to W hours either way, or only earlier. A reach of one hour either way lifts the 2025 saving share from 90.7% to 93.1%, or from 4.05 to 4.16 million EUR a year against the standard shift, winning back about 105,000 of the 233,000 EUR a year that fixing the week costs. The start moves on about a third of days, and wider reaches add little (93.8% at four hours). On the validation years the gain is smaller, about 74,000 of a 438,000 EUR gap at one hour. Moving earlier only wins back roughly half as much as moving either way.

With the 11-hour rest rule kept, the planned-then-adjusted schedule does better than choosing each day freely the day before: 4.04 million EUR a year at a one-hour reach and 4.08 million at four hours, against 4.01 million on 2025, and 3.56 and 3.61 million against 3.29 million on the validation years. Choosing one day at a time under the rule, a late shift today can rule out a cheap early shift tomorrow; a plan that already keeps the rule holds each day's choice near a sequence that works. Under the law, a week's notice with one hour of room costs the plant nothing.

| Reach | Saving share 2025, free | EUR a year 2025, free | Saving share 2025, rest kept | EUR a year 2025, rest kept | Saving share validation, free | Saving share validation, rest kept | Days moved 2025, free |
|---|---|---|---|---|---|---|---|
| Week-ahead plan (0 h) | 90.7% | 4,055,000 | 89.8% | 3,926,000 | 86.1% | 89.9% | 0% |
| 1 h either way | 93.1% | 4,160,000 | 92.5% | 4,045,000 | 87.9% | 92.2% | 35.1% |
| 2 h either way | 93.3% | 4,172,000 | 92.8% | 4,057,000 | 88.1% | 92.5% | 35.1% |
| 3 h either way | 93.5% | 4,177,000 | 92.8% | 4,060,000 | 88.5% | 93.0% | 35.9% |
| 4 h either way | 93.8% | 4,191,000 | 93.3% | 4,078,000 | 88.9% | 93.3% | 36.4% |
| 1 h earlier only | 91.8% | 4,104,000 | 91.1% | 3,984,000 | 86.7% | 91.0% | 12.9% |
| 2 h earlier only | 92.0% | 4,112,000 | 91.4% | 3,995,000 | 86.9% | 91.5% | 12.9% |
| 3 h earlier only | 92.1% | 4,115,000 | 91.4% | 3,996,000 | 87.1% | 92.0% | 12.9% |
| 4 h earlier only | 92.3% | 4,125,000 | 91.8% | 4,012,000 | 87.3% | 92.4% | 13.2% |
| Day-ahead choice (any start) | 95.9% | 4,287,000 | 91.7% | 4,009,000 | 96.7% | 85.3% | |

Saving share is `(standard - shift_cost) / (standard - hindsight)`; with rest kept, hindsight is the cheapest rest-keeping sequence over the year, found by dynamic programming. EUR a year is the saving against the standard 06:00 to 14:00 shift for 500 MWh a day. Validation is the mean of 2023 and 2024, each scored on its own. With rest kept, the plan is the week-ahead schedule with rest kept, and the final start must also keep 11 hours after the previous day's final shift; such a start always exists, since the planned start plus the later reach is never too early.
