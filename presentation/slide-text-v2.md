# DBA3803 group presentation v2: slide text and speaker notes

Text of `dba3803-group-presentation-v2.pptx`, the visual-first version (15 slides, three per speaker, native equations), extracted 2026-10-24. What came off the slides is in the speaker notes. Every number comes from the notebook in `Programming/dba-group-project/`.

<!-- Slide number: 1 -->
DBA3803 · PREDICTIVE ANALYTICS · GROUP PROJECT

### Chart: Thu 6 March 2025 · price + premium, EUR/MWh

| Category | Cheapest shift | Other hours |
| --- | --- | --- |
| 00 | 0.0 | 93.9 |
| 01 | 0.0 | 94.8 |
| 02 | 0.0 | 97.0 |
| 03 | 0.0 | 98.1 |
| 04 | 0.0 | 104.6 |
| 05 | 0.0 | 105.9 |
| 06 | 0.0 | 139.3 |
| 07 | 0.0 | 151.3 |
| 08 | 108.5 | 0.0 |
| 09 | 78.7 | 0.0 |
| 10 | 50.5 | 0.0 |
| 11 | 4.0 | 0.0 |
| 12 | 0.1 | 0.0 |
| 13 | 0.0 | 0.0 |
| 14 | 13.2 | 0.0 |
| 15 | 71.0 | 0.0 |
| 16 | 0.0 | 116.5 |
| 17 | 0.0 | 148.9 |
| 18 | 0.0 | 165.4 |
| 19 | 0.0 | 149.1 |
| 20 | 0.0 | 125.6 |
| 21 | 0.0 | 106.6 |
| 22 | 0.0 | 111.5 |
| 23 | 0.0 | 101.1 |

When should the factory switch on?
Forecasting the German day-ahead power price to pick the cheapest 8-hour shift, every day.
[Presenter names]
Highlighted: that day's cheapest shift, 08:00–16:00. The model picked it too.
Data: Energy-Charts API, January 2019 to October 2026

### Notes:
Speaker 1. Open with the question: a factory has to pick when tomorrow's 8-hour shift runs, and the price of power changes every hour. We forecast the German day-ahead price so it can pick the cheapest shift every day, before the auction closes. The chart is Thursday 6 March 2025. We averaged the plain price over all 24 hours of every 2025 day, took the median of those 365 averages, and chose the weekday closest to it. So its price level is typical, and the day was chosen without looking at how the model did. That day the cheapest 8 hours, 08:00 to 16:00, sit on the midday solar dip, and our model picked exactly those hours. An 08:00 start is not the usual winner, though: across 2019 to 2023 it was midnight or 09:00 to 10:00 (slide 5). Everything in this talk is judged by how much money the shift choice saves.

<!-- Slide number: 2 -->
01 · CONTEXT
Power is priced hour by hour, and sometimes below zero
1,018

### Chart: Yearly mean day-ahead price, EUR/MWh

| Category | Crisis year | Other years |
| --- | --- | --- |
| 2019 | 0.0 | 37.7 |
| 2020 | 0.0 | 30.5 |
| 2021 | 0.0 | 96.9 |
| 2022 | 235.4 | 0.0 |
| 2023 | 0.0 | 95.2 |

hours below zero, 2019 to 2023
−500 to 871
EUR/MWh, lowest and highest hour
Almost 8×
the yearly mean, from 2020 to 2022
Speaker 1 · Context and decision
2

### Notes:
Prices swing within every day, sometimes below zero: 1,018 negative hours in five years, mostly at midday (solar) and at night (wind with low demand). Wind and solar outran demand, so buyers were paid to take power. The lowest hour was −500 EUR/MWh in 2023, the highest 871 in the 2022 gas crisis. And the level moved: the yearly mean went from 30 EUR/MWh in 2020 to 235 in 2022, almost 8 times as high, and never came back down. 2023 still averaged 95, two and a half times 2019. Our data has no fuel prices, so the model has to learn the level from recent prices. That becomes important on slide 8.

<!-- Slide number: 3 -->
02 · THE DECISION
Each morning the factory picks tomorrow's start hour
D-1 · 11:55
D-1 · 12:00
Day D
Pick the start
Auction clears
Run the shift
00:00 to 16:00
a price for every hour
8 h × 62.5 MW = 500 MWh
Outlined = +10 EUR/MWh night premium
s = start hour, 00 to 16
00
01
02
03
04
05
06
07
08
09
10
11
12
13
14
15
16
17
18
19
20
21
22
23
standard 06:00–14:00
cheapest on 6 March 2025
Speaker 1 · Context and decision
3

### Notes:
Every input the model uses must be known at 11:55, five minutes before the day-ahead auction closes, and that one constraint shapes the rest of the project. Any start from 00:00 to 16:00 is allowed, so 17 options, and the auction then sets one clearing price for every hour of tomorrow. The shift runs 8 hours at 62.5 MW, so 500 MWh. Its cost is the average of price plus premium over those 8 hours, from the start hour s to s plus 7. The premium is 625 EUR an hour of staff cost spread over 62.5 MWh, so 10 EUR/MWh, at night and all weekend. That is what makes the choice hard: a night hour only pays if it is more than 10 EUR/MWh cheaper than the day hour it replaces. Savings are measured against the standard 06:00 to 14:00 shift.

<!-- Slide number: 4 -->
02 · THE DECISION
Only the shape of the day matters, not its level

### Chart: Share of price movement, by year

| Category | Shape: hours within the day | Level: whole day up or down |
| --- | --- | --- |
| 2019 | 0.418 | 0.5820000000000001 |
| 2020 | 0.367 | 0.633 |
| 2021 | 0.175 | 0.825 |
| 2022 | 0.194 | 0.806 |
| 2023 | 0.431 | 0.569 |

+20 EUR/MWh on every hour
→ same cheapest shift
Speaker 2 · Exploration and data
4

### Notes:
Speaker 2. The whole project rests on this idea. Split every price into the day's mean, the level, and the hour's distance from it, the shape. Move every hour up by the same amount and every shift gets more expensive by the same amount, so the ranking stays the same: a forecast 20 EUR/MWh too high in every hour still picks the right shift. The decision only uses the shape. Shape share is the within-day variance divided by the total. In calm years the shape is about 40% of the movement; in 2021 and 2022 it is under 20%, so over 80% was whole days moving together. A model that minimises error over all hours spends most of its effort on the level, which the decision never uses. That is why we score models on the cost of the shift they pick, with MAE and RMSE reported beside it.

<!-- Slide number: 5 -->
03 · EXPLORATION
The cheapest shift starts at midnight or mid-morning

| 2019 | 33% |  |  |  |  |  |  | – |  | 15% | 25% |  |  |  |  | – |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 31% |  |  |  | – | – |  | – |  | 17% | 27% |  |  |  |  | – |  |
| 2021 | 38% |  |  |  | – | – |  | – |  | 14% | 29% |  |  | – |  | – |  |
| 2022 | 39% |  | – |  | – | – |  | – |  | 22% | 24% |  |  | – | – | – |  |
| 2023 | 30% |  |  |  |  | – |  |  |  | 24% | 33% |  | – | – |  | – |  |
|  | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |

Share of days each start hour was cheapest, premium included (hindsight). A dash means never.
00:00
09:00–10:00
06:00, standard
30–39% of days
40–57% of days
1% at most
Speaker 2 · Exploration and data
5

### Notes:
Hindsight on the training years, 2019 to 2023: for each day, which start was actually cheapest, premium included. Only shares of 10% or more are labelled. Two starts win most days: midnight, on 30 to 39% of days, when demand is low and wind often blows and night hours are cheap enough to cover the 10 EUR/MWh premium; and 09:00 or 10:00, on 40 to 57% together, when the shift sits on the midday solar dip. Neither wins every day, so the start has to be picked daily. The standard 06:00 start, the dashed column, was cheapest on 1% of days at most, and it is the benchmark every saving is measured against.

<!-- Slide number: 6 -->
04 · DATA
Seven years of hourly prices and forecasts
68,183
German price
target
one row
per delivery hour
Grid forecasts
hours, Jan 2019 to Oct 2026
load · solar · wind
11 neighbour prices
tested, dropped
the demand left for gas and coal
Speaker 2 · Exploration and data
6

### Notes:
68,183 hours from the Energy-Charts API, 1 January 2019 to 11 October 2026. The target is the DE-LU day-ahead clearing price. The inputs are the grid operators' forecasts of load, solar, onshore and offshore wind. The auction moved to 15-minute prices in October 2025, so everything is averaged to hourly. From the forecasts we build residual load, load minus solar minus wind: the demand left for gas and coal. The extremes, the −500 hour and the 2022 spikes, are kept, because they are actual auction results, not data errors. Every feature must be known at 11:55 the day before. Tomorrow's prices are not, so all price features are lags of 24 hours or more. The data holds no actual load or wind, only forecasts, which removes the most common leak. The 11 neighbouring zones move almost one for one with Germany in the same hour, but that is a leak, since they clear in the same auction. A day back, the best one correlates 0.906 with the German price, against 0.903 for Germany's own, so they add nothing and were dropped.

<!-- Slide number: 7 -->
05 · FEATURES
17 features, all known by 11:55 the day before
4
5
8
Calendar
Fundamentals
Price memory
hour
weekday
month
weekend
load
solar
wind on
wind off
lag 24 h
lag 48 h
lag 168 h
residual
D-1 mean
D-1 min
D-1 max
7-day mean
30-day mean
WHAT IS KNOWN AT THE 11:55 GATE
Past prices
through end of D-1
Grid forecasts
for day D *
Calendar
any date
11:55 gate
D 00:00
* published 18:00 on D-1, so a buyer uses a vendor forecast
Speaker 3 · Features and models
7

### Notes:
Speaker 3. Three groups, each from a finding in the exploration. Calendar because prices follow the clock and the week: Sundays sit 27 EUR/MWh below their week, and solar moved the cheapest hours from 03:00 to 13:00. Fundamentals because residual load sets the shape of the day: its within-day correlation with the price is 0.70, against at most 0.29 for any single forecast. Price memory because recent prices carry the level, which our data cannot measure directly: same hour yesterday is the best single guide, correlation 0.90, and the 168-hour lag adds the weekly cycle. The timeline shows what is known at the gate. One input is a stand-in: the grid operators publish their forecasts at 18:00, after the gate, so in practice the buyer needs a vendor forecast of similar quality, made the same morning.

<!-- Slide number: 8 -->
05 · FEATURES
Recent prices tell the model what kind of market it is in

### Chart: EUR/MWh added per extra GW of residual load

| Category | EUR/MWh per GW |
| --- | --- |
| 2019 | 1.1 |
| 2020 | 1.2 |
| 2021 | 3.5 |
| 2022 | 7.1 |
| 2023 | 3.1 |

### Chart: Saving share as feature groups are added

| Category | Kept | Dropped |
| --- | --- | --- |
| Calendar + forecasts | 64.4 | 0.0 |
| + price memory | 95.8 | 0.0 |
| + residual × level | 0.0 | 96.2 |
| + neighbour spreads | 0.0 | 95.6 |

Speaker 3 · Features and models
8

### Notes:
Left: the same extra GW of residual load added 1.1 EUR/MWh in 2019 and 7.1 in 2022, because the gas price sets what the marginal plant costs. Pooled over all years, the correlation between residual load and price drops to 0.39 for this reason. With no fuel prices in the data, recent prices are how the model learns the level. Right, Lasso into XGBoost scored on the mean of the 2023 and 2024 folds: without price memory the model cannot tell 2019 from 2022 and reaches only 64% of the saving. Adding price memory lifts it to 96%. A group stays only if it wins by more than the year-to-year swing. The two extra ideas, residual load times the price level and the price gaps to the neighbours, moved the score by less than that, so they were dropped, since the trees build the interaction on their own.

<!-- Slide number: 9 -->
06 · MODELS
Three models, measured against two simple baselines
MODELS
BASELINES · SHARE OF THE POSSIBLE SAVING, 2019–23
87–97%
XGBoost
CHOSEN
boosted decision trees
Residual-load rule
the 8 hours with the lowest forecast residual load
Lasso → XGBoost
linear base, trees fix the rest
52–71%
LEAR
Same hour yesterday
research benchmark, one Lasso per hour
tomorrow repeats today
Speaker 3 · Features and models
9

### Notes:
The rule is the most important box on this slide. Picking the 8 hours with the lowest forecast residual load needs no price model at all and already gets 87% to 97% of the possible saving. So any model can add at most the last few percent, and only by doing what the rule cannot: weighing the 10 EUR/MWh night premium in euros. Same hour yesterday, assuming tomorrow repeats today, is the usual yardstick for price error but weak at picking shifts, 52 to 71%. The models: XGBoost is boosted decision trees, which follow bends a straight line cannot, like the price jump once residual load passed about 52 GW in 2021, and let the residual-load effect grow with the price level. Lasso into XGBoost fits the steady daily and weekly pattern with a Lasso, then trees correct what it gets wrong; it tests whether the linear stage adds anything. LEAR is the standard benchmark in price-forecasting research (Lago et al. 2021): one Lasso per delivery hour on asinh-scaled prices, so beating it matters. Left out: Lasso alone, a neural network (costly, and nothing pointed to a pattern the trees would miss), and one model per hour, which LEAR already tests. Hand over to how we scored them.

<!-- Slide number: 10 -->
06 · MODELS
Always train on the past, score on a later year
2019
2020
2021
2022
2023
2024
2025
2026
Fold 1
Fold 2
Final model
Reserve check
train
validate
test, once
reserve
SCORE
ERROR, COMPARABLE ACROSS YEARS
A lead counts only if it beats the year-to-year swing. On a tie, the simpler model wins.
Speaker 4 · Evaluation and results
10

### Notes:
Speaker 4. Every model and setting is chosen on two validation years, 2023 and 2024, each predicted by a model trained only on the years before it. The folds cover three markets: calm years, the gas crisis and the recovery. Only the final model ever sees 2025, once, and January to October 2026 is held back as a reserve. Fold 1 is slightly optimistic, because the exploration looked at 2023. The score is the saving share: 100% means the cheapest shift every day, 0% means no better than always starting at 06:00. It comes from each year's totals, so a day with more to save counts for more. MASE scales the error by same hour yesterday's, which misses by 9 EUR/MWh in 2019 but 58 in 2022, so calm and crisis years compare. To choose between two models: a model only beats another if its saving share leads by more than the swing between the two years, the fold SD; inside that, the lower MAE wins if it leads by more than its own fold SD; if both tie, the simpler model wins.

<!-- Slide number: 11 -->
07 · RESULTS
Both tree models beat the rule and tie with each other
17.1

### Chart: Saving share, mean of the 2023 and 2024 folds

| Category | Model | Rule |
| --- | --- | --- |
| Lasso → XGBoost · MAE 17.5 | 95.6 | 0.0 |
| XGBoost · MAE 18.9 | 95.4 | 0.0 |
| Residual-load rule | 0.0 | 94.1 |
| LEAR · MAE 17.1 | 93.5 | 0.0 |
| Same hour yesterday · MAE 27.5 | 73.7 | 0.0 |

LEAR's MAE, the lowest, yet last of the three on saving share
0.2 pts
between the two trees, inside a swing of 1.3, so a tie
XGBoost goes forward
depth 3 · 200 trees · learning rate 0.05
Speaker 4 · Evaluation and results
11

### Notes:
Both tree models beat the rule, by 1.3 to 1.5 points. Fold SDs: Lasso into XGBoost 1.3, XGBoost 0.3, LEAR 1.2 points. LEAR has the lowest error of all three, MAE 17.1 EUR/MWh, but it is 2.1 points behind Lasso into XGBoost on saving share, more than the swing between years: pricing each hour well is not the same as ranking the hours, and ranking is what picks the shift. Between the two tree models the gap is 0.2 points against a swing of 1.3, and the MAE gap is inside its swing too. A tie on both counts sends the simpler XGBoost forward. The chart shows default settings. After tuning it is still a tie, 96.5% against 95.6% inside a fold SD of 1.5, and a random search of 100 settings did not change that. The final model is XGBoost with depth 3, 200 trees and a learning rate of 0.05.

<!-- Slide number: 12 -->
07 · RESULTS · TEST YEAR
On 2025, a year it never saw, it saves 4.31M EUR
96.4%

### Chart: Saved in 2025 against the standard 06:00 shift, million EUR

| Category | Hindsight | Model | Rule | Others |
| --- | --- | --- | --- | --- |
| Hindsight (best) | 4.47 | 0.0 | 0.0 | 0.0 |
| XGBoost forecast | 0.0 | 4.31 | 0.0 | 0.0 |
| Residual-load rule | 0.0 | 0.0 | 4.2 | 0.0 |
| Same hour yesterday | 0.0 | 0.0 | 0.0 | 3.72 |
| Last year's profile | 0.0 | 0.0 | 0.0 | 3.05 |

of the best possible saving (rule: 94.1%)
+105k EUR
a year more than the rule
71.5% vs 70.7%
days on the exact cheapest shift, model vs rule
Speaker 4 · Evaluation and results
12

### Notes:
The final model was refitted on 2019 to 2024 and scored once on 2025. No other model touched 2025. It captures 96.4% of the possible saving, against 94.1% for the rule: 4.31 million EUR against 4.20 million, for 500 MWh a day, so about 105,000 EUR a year more. The lead over the rule was 1.5 points in validation and 2.3 points here, so it held. Both pick the exact cheapest shift about equally often, 71.5% against 70.7% of days, so the edge comes from the days each one misses: the model's misses cost less. The 2026 reserve, January to October, scored once, agrees with a smaller lead: 98.7% against 97.8%, about 49,000 EUR a year.

<!-- Slide number: 13 -->
07 · RESULTS
Is the 105k EUR edge more than luck?
MODEL'S YEARLY LEAD OVER THE RULE, THOUSAND EUR · 95% INTERVAL OF 10,000 RESAMPLES
105k
7-day blocks
8k
223k
30-day blocks
−1.5k
239k
−50
0
50
100
150
200
250
Beats the rule in more than 97% of resamples. How big the edge is stays uncertain.
Speaker 5 · Robustness and recommendation
13

### Notes:
Speaker 5. The question a sceptic asks: could 105,000 EUR be luck in which days 2025 happened to bring? We compared the model and the rule on the same 365 days and resampled those days 10,000 times, in runs of consecutive days, because neighbouring days are alike through the weekly cycle and spells of weather. With 7-day runs the whole 95% interval is above zero, from about 8,000 to 223,000 EUR a year. [Click] With 30-day runs, which keep longer stretches of similar days together, the low end just touches zero, at about −1,500 EUR. So the model beats the rule in more than 97% of resamples, but the edge could be anything from almost nothing to more than twice 105,000 EUR. This only covers luck within 2025; whether another year would favour the model as much rests on two validation years.

<!-- Slide number: 14 -->
08 · DIAGNOSTICS
It gets the shape right, but misses the spikes

### Chart: Thu 6 March 2025 · price + premium, EUR/MWh

| Category | Model's shift = cheapest shift | Cleared | Forecast |
| --- | --- | --- | --- |
| 00 | 0.0 | 93.9 | 117.2 |
| 01 | 0.0 | 94.8 | 114.1 |
| 02 | 0.0 | 97.0 | 114.1 |
| 03 | 0.0 | 98.1 | 116.1 |
| 04 | 0.0 | 104.6 | 119.2 |
| 05 | 0.0 | 105.9 | 123.8 |
| 06 | 0.0 | 139.3 | 150.0 |
| 07 | 0.0 | 151.3 | 167.4 |
| 08 | 200.0 | 108.5 | 135.3 |
| 09 | 200.0 | 78.7 | 99.1 |
| 10 | 200.0 | 50.5 | 79.5 |
| 11 | 200.0 | 4.0 | 70.1 |
| 12 | 200.0 | 0.1 | 59.3 |
| 13 | 200.0 | 0.0 | 62.5 |
| 14 | 200.0 | 13.2 | 72.5 |
| 15 | 200.0 | 71.0 | 95.4 |
| 16 | 0.0 | 116.5 | 141.6 |
| 17 | 0.0 | 148.9 | 190.8 |
| 18 | 0.0 | 165.4 | 195.8 |
| 19 | 0.0 | 149.1 | 191.6 |
| 20 | 0.0 | 125.6 | 165.4 |
| 21 | 0.0 | 106.6 | 147.6 |
| 22 | 0.0 | 111.5 | 145.1 |
| 23 | 0.0 | 101.1 | 131.9 |

### Chart: MAE by price band, EUR/MWh, 2023–24 out of fold

| Category | MAE |
| --- | --- |
| below 0 (758 h) | 16.5 |
| 0 to 200 (16,545 h) | 15.9 |
| above 200 (241 h) | 89.1 |

93.1% vs 95.6%
predicting asinh(price) to shrink the spikes: worse, rejected
Speaker 5 · Robustness and recommendation
14

### Notes:
Left: the day from slide 1, Thursday 6 March 2025, the 2025 weekday with the median price level. The forecast runs 11 to 66 EUR/MWh too high in every hour and misses how deep the solar dip goes, but the shape is right, so its shift is the cheapest one. That is the whole argument in one picture: poor MAE on this day, perfect decision. Right, on out-of-fold predictions for 2023 and 2024: ordinary and negative hours are missed by about 16 EUR/MWh, but the 241 hours above 200 by 89, and on average they are predicted 61 EUR/MWh too low. That is the model's weak point. The one fix the diagnostics suggested, predicting asinh of the price, cut the saving share from 95.6% to 93.1%, because shrinking the big prices also shrinks the gaps between hours, and those gaps pick the shift. So the model stays as it is and the spikes stay a known weakness.

<!-- Slide number: 15 -->
WRAP-UP
Schedule by forecast. Most of the money is in choosing at all
4.2M EUR
LIMITS
Forecast timing  inputs arrive at 18:00, after the gate
a year from choosing the shift at all, rule included
Thin evidence  two validation years, one test year
+105k EUR
Spikes  hours above 200 EUR/MWh missed by 89
a year more from the model, on the same bought forecast
Simplified decision  one-day shifts, plant does not move the price
Price memory
Thank you. Questions?
the feature group that matters
Speaker 5 · Robustness and recommendation
15

### Notes:
Most of the money comes from choosing the shift at all, not from the model: the rule alone is worth 4.2 million EUR a year against the standard 06:00 shift, about 40 times the forecast's own gain. Both options need a vendor forecast of load, solar and wind, because the rule ranks the hours by forecast residual load, which comes from that same forecast. So buying it is a cost of choosing the shift at all, not of the model. On top of that forecast, the model adds about 105,000 EUR a year for the cost of running it on public price history, and 2026 agrees with a smaller lead. Keep the rule as the fallback on days the model fails. Limitations, ranked: the grid operators publish their forecasts at 18:00, after the gate, so the rule and the model both assume a bought vendor forecast; two validation years and one test year make the lead over the rule a narrow call; spikes are missed; the decision is simplified, with shifts inside one day, premiums that do not stack and no public holidays or fuel prices in the data, and LEAR refitted yearly rather than daily. On the price-taker point: buying adds about 0.18 EUR/MWh to the price paid, under 2% of what the forecast saves. Everyone should be ready to take questions on any slide.
