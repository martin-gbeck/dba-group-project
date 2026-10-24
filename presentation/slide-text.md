# DBA3803 group presentation: slide text and speaker notes

Text of `dba3803-group-presentation.pptx` (15 slides, three per speaker), extracted from the deck on 2026-10-24. Every number comes from the notebook in `Programming/dba-group-project/`.

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
Forecasting the German day-ahead power price to pick the cheapest 8-hour shift, every day, before the auction closes.
[Presenter names]
A 2025 weekday with a median price level. The highlighted bars, 08:00 to 16:00, were that day's cheapest shift, and the model picked the same one.
Data: Energy-Charts API, January 2019 to October 2026

### Notes:
Speaker 1. Open with the question: a factory has to pick when tomorrow's shift runs, and the price of power changes every hour. The chart is Thursday 6 March 2025. We averaged the plain price over all 24 hours of every 2025 day, took the median of those 365 averages, and chose the weekday closest to it. So its price level is typical, and the day was chosen without looking at how the model did. That day the cheapest 8 hours, 08:00 to 16:00, sit on the midday solar dip, and our model picked exactly those hours. An 08:00 start is not the usual winner, though: across 2019 to 2023 it was midnight or 09:00 to 10:00 (slide 5). Everything in this talk is judged by how much money the shift choice saves.

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

hours with a negative price, 2019 to 2023. Wind and solar outran demand, so buyers were paid to take power.
−500 to 871
EUR/MWh: the lowest hour (2023) and the highest (2022).
Almost 8×
the yearly mean price, from 30 EUR/MWh in 2020 to 235 in the 2022 gas crisis.
The level never went back to where it was before the crisis: 2023 still averaged 95 EUR/MWh, two and a half times 2019.
Speaker 1 · Context and decision
2

### Notes:
Prices swing within every day, sometimes below zero: 1,018 negative hours in five years, mostly at midday (solar) and at night (wind with low demand). And the level moved: the yearly mean went from 30 EUR/MWh in 2020 to 235 in 2022, and never came back down. Our data has no fuel prices, so the model has to learn the level from recent prices. That becomes important on slide 8.

<!-- Slide number: 3 -->
02 · THE DECISION
Each morning the factory picks tomorrow's start hour
D-1 · 11:55
D-1 · 12:00
Day D
Pick the start hour
Auction closes
Run the shift
Any start from 00:00 to 16:00, so 17 options. Locked five minutes before the auction closes.
One clearing price is set for every hour of tomorrow. Until then, the prices are unknown.
8 hours at 62.5 MW, so 500 MWh bought at those hours' prices.
ONE WEEKDAY, 24 HOURS
Outlined hours carry the +10 EUR/MWh night premium
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
cheapest on 6 March 2025: 08:00–16:00
Shift cost = average of (price + premium) over the 8 hours. The premium is 625 EUR an hour of staff cost spread over 62.5 MWh, at night and all weekend.
Speaker 1 · Context and decision
3

### Notes:
Every input the model uses must be known at 11:55, five minutes before the day-ahead auction closes, and that one constraint shapes the rest of the project. The night and weekend premium is what makes the choice hard: a night hour only pays if it is more than 10 EUR/MWh cheaper than the day hour it replaces.

<!-- Slide number: 4 -->
02 · THE DECISION
Only the shape of the day matters, not its level
Move every hour up by the same amount and every shift gets more expensive by the same amount. The ranking stays the same.

### Chart: Share of price movement, by year

| Category | Shape: hours within the day | Level: whole day up or down |
| --- | --- | --- |
| 2019 | 0.418 | 0.5820000000000001 |
| 2020 | 0.367 | 0.633 |
| 2021 | 0.175 | 0.825 |
| 2022 | 0.194 | 0.806 |
| 2023 | 0.431 | 0.569 |

A forecast 20 EUR/MWh too high in every hour still picks the right shift.
So models are judged on the cost of the shift they pick. MAE and RMSE are reported beside it.
Each hour's price = its day's mean (level) + its distance from that mean (shape). In 2021 and 2022 over 80% of the movement was whole days moving together.
Speaker 2 · Exploration and data
4

### Notes:
Speaker 2. The whole project rests on this idea. Split every price into the day's mean and the hour's distance from it. The decision only uses the distance, the shape. Shape share = within-day variance divided by total variance. In calm years the shape is about 40% of the movement, in the crisis years under 20%. So a model that minimises error over all hours spends most of its effort on the level, which the decision never uses. That is why we score models on the shift they pick.

<!-- Slide number: 5 -->
03 · EXPLORATION
The cheapest shift starts at midnight or mid-morning

| 2019 | 33% |  |  |  |  |  |  |  |  | 15% | 25% |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 31% |  |  |  |  |  |  |  |  | 17% | 27% |  |  |  |  |  |  |
| 2021 | 38% |  |  |  |  |  |  |  |  | 14% | 29% |  |  |  |  |  |  |
| 2022 | 39% |  |  |  |  |  |  |  |  | 22% | 24% |  |  |  |  |  |  |
| 2023 | 30% |  |  |  |  |  |  |  |  | 24% | 33% |  |  |  |  |  |  |
|  | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |

Share of days each start hour was cheapest, premium included, 2019 to 2023 (hindsight, actual prices). Only shares of 10% or more are labelled; a dash means never cheapest.
00:00 on 30–39% of days. Night hours are cheap enough to cover the 10 EUR/MWh premium.
09:00–10:00 on 40–57%. The shift sits on the midday solar dip.
06:00, the usual early shift, on 1% at most. That is the benchmark the savings are measured against.
Speaker 2 · Exploration and data
5

### Notes:
Hindsight on the training years: for each day, which start was actually cheapest, premium included. Two starts win most days: midnight, when demand is low and wind often blows, and mid-morning, when the shift covers the solar dip. Neither wins every day, so the start has to be picked daily. The normal 06:00 start is almost never the cheapest, which is why picking it daily is worth so much.

<!-- Slide number: 6 -->
04 · DATA
Seven years of hourly prices and forecasts
TARGET
INPUTS
TESTED
German price
Grid forecasts
11 neighbours
DE-LU day-ahead clearing price, EUR/MWh, one row per delivery hour.
The grid operators' forecasts of load, solar, onshore and offshore wind.
Prices in France, the Netherlands, Austria and 8 other connected zones.
68,000+
Averaged to hourly. The auction went to 15-minute prices in October 2025.
Built residual load = load − solar − wind: the demand left for gas and coal.
hours, 1 January 2019 to 11 October 2026, from the Energy-Charts API.
Kept the extremes. The −500 hour and the 2022 spikes are actual auction results, not data errors.
Dropped the neighbours. A day back, the best one correlates 0.906 with the German price, against 0.903 for Germany's own.
Speaker 2 · Exploration and data
6

### Notes:
Every feature must be known at 11:55 the day before. Tomorrow's prices are not, so all price features are lags of 24 hours or more. The data holds no actual load or wind, only forecasts, which removes the most common leak. Neighbouring zones move almost one for one with Germany in the same hour, but that is a leak, since they clear in the same auction. A day back they tell us nothing Germany's own price does not, so they were dropped.

<!-- Slide number: 7 -->
05 · FEATURES
17 features, all known by 11:55 the day before
4 · CALENDAR
5 · FUNDAMENTALS
8 · PRICE MEMORY
hour · weekday · month · weekend
load · solar · wind on · wind off · residual
lag 24, 48, 168 h · D-1 mean, min, max · 7-day and 30-day means
Sundays sit 27 EUR/MWh below their week. Solar moved the cheapest hours from 03:00 to 13:00.
Residual load drives the shape: within-day correlation 0.70, against at most 0.29 for any single forecast.
Same hour yesterday is the best single guide (correlation 0.90). Lag 168 adds the weekly cycle.
WHAT IS KNOWN AT THE 11:55 GATE
Past prices
through end of D-1
Grid forecasts
for day D *
Calendar
any date
D-1 00:00
11:55 gate
D 00:00
* The grid operators publish these at 18:00 on D-1. In practice a buyer would pay for a vendor forecast made the same morning.
Speaker 3 · Features and models
7

### Notes:
Speaker 3. Three groups, each from a finding in the exploration. Calendar because prices follow the clock and the week. Fundamentals because residual load sets the shape of the day. Price memory because recent prices carry the level, which our data cannot measure directly. One input is a stand-in: the grid operators publish their forecasts at 18:00, after the gate, so in practice the buyer needs a vendor forecast of similar quality.

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
| + price memory (kept) | 95.8 | 0.0 |
| + residual × level (dropped) | 0.0 | 96.2 |
| + neighbour spreads (dropped) | 0.0 | 95.6 |

With no fuel prices in the data, recent prices are how the model learns the level.
Lasso-then-XGBoost, mean of the 2023 and 2024 folds. A group stays only if it wins by more than the year-to-year swing.
Speaker 3 · Features and models
8

### Notes:
Left: the same extra GW of residual load added 1.1 EUR/MWh in 2019 and 7.1 in 2022, because the gas price sets what the marginal plant costs. Pooled over all years, the correlation between residual load and price drops to 0.39 for this reason. Right: without price memory the model cannot tell 2019 from 2022, and it reaches only 64% of the saving. Adding price memory lifts it to 96%. The two extra ideas, residual load times the price level and the price gaps to the neighbours, moved the score by less than the swing between years, so they were dropped, since the trees build the interaction on their own.

<!-- Slide number: 9 -->
06 · MODELS
Three models, measured against two simple baselines
XGBoost
Boosted decision trees. They follow bends a straight line cannot, like the price jump once residual load passed about 52 GW in 2021, and let the residual-load effect grow with the price level.
CHOSEN
Lasso → XGBoost
A Lasso fits the steady daily and weekly pattern, then trees correct what it gets wrong. Tested to see whether the linear stage adds anything.
LEAR
The standard benchmark in price-forecasting research (Lago et al. 2021): one Lasso per delivery hour on asinh-scaled prices, using the last few days' prices and forecasts.
Residual-load rule
No price model: run the 8 hours with the lowest forecast residual load. Already gets 87–97% of the saving, but cannot see the night premium.
BASELINE
Same hour yesterday
Assume tomorrow repeats today. The usual yardstick for price error, but weak at picking shifts: 52–71% of the saving.
BASELINE
Left out: Lasso alone (the same model without the trees), a neural network (costly, and nothing pointed to a pattern the trees would miss), one model per hour (LEAR already tests that).
Speaker 3 · Features and models
9

### Notes:
The rule is the most important line on this slide. Picking the 8 hours with the lowest forecast residual load needs no price model at all and already gets 87% to 97% of the possible saving. So any model can add at most the last few percent, and only by doing what the rule cannot: weighing the 10 EUR/MWh night premium in euros. LEAR is the method the research field uses as its standard, so beating it matters. Hand over to how we scored them.

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
reserve, Jan–Oct 2026
SCORE
RULES
ERROR
Saving share
Set in advance
MASE beside MAE
(standard − chosen) ÷ (standard − hindsight). 100% = the cheapest shift every day; 0% = no better than always starting at 06:00.
A model must win by more than the year-to-year swing (fold SD). On a tie, the simpler model goes forward.
A naive forecast misses by 9 EUR/MWh in 2019 but 58 in 2022. Scaling by it makes calm and crisis years comparable.
Speaker 4 · Evaluation and results
10

### Notes:
Speaker 4. Every model and setting is chosen on two validation years, 2023 and 2024, each predicted by a model trained only on the years before it. The folds cover three markets: calm years, the gas crisis and the recovery. Only the final model ever sees 2025, once. Fold 1 is slightly optimistic, because the exploration looked at 2023. The saving share comes from each year's totals, so a day with more to save counts for more. To choose between two models: a model only beats another if its saving share leads by more than the swing between the two years; inside that, the lower MAE wins if it leads by more than its own fold SD; if both tie, the simpler model wins.

<!-- Slide number: 11 -->
07 · RESULTS
Both tree models beat the rule, so the simpler one goes forward
LEAR prices best, ranks worst

### Chart: Saving share, mean of the 2023 and 2024 folds

| Category | Model | Rule |
| --- | --- | --- |
| Lasso → XGBoost · MAE 17.5 | 95.6 | 0.0 |
| XGBoost · MAE 18.9 | 95.4 | 0.0 |
| Residual-load rule | 0.0 | 94.1 |
| LEAR · MAE 17.1 | 93.5 | 0.0 |
| Same hour yesterday · MAE 27.5 | 73.7 | 0.0 |

Lowest error (MAE 17.1 EUR/MWh) but 2.1 points behind Lasso → XGBoost on saving share, more than the swing between years. Ranking the hours is what picks the shift.
After tuning: still a tie
Lasso → XGBoost 96.5% against XGBoost 95.6%, inside the fold SD of 1.5. XGBoost goes forward as the simpler model: depth 3, 200 trees, learning rate 0.05.
Default settings in the chart; tuned settings on the right. Fold SDs: Lasso → XGBoost 1.3, XGBoost 0.3, LEAR 1.2 points.
Speaker 4 · Evaluation and results
11

### Notes:
Both tree models beat the rule, by 1.3 to 1.5 points. LEAR has the lowest error of all three but the lowest saving share: pricing each hour well is not the same as ranking the hours. Between the two tree models, the gap is 0.2 points against a swing of 1.3, and the MAE gap is inside its swing too. A tie on both counts sends the simpler XGBoost forward. Tuning both and a random search of 100 settings did not change that.

<!-- Slide number: 12 -->
07 · RESULTS · TEST YEAR
On 2025, a year it never saw, it saves 4.31M EUR
96.4%

### Chart: Saved in 2025 against a fixed 06:00 start, million EUR

| Category | Hindsight | Model | Rule | Others |
| --- | --- | --- | --- | --- |
| Hindsight (best) | 4.47 | 0.0 | 0.0 | 0.0 |
| XGBoost forecast | 0.0 | 4.31 | 0.0 | 0.0 |
| Residual-load rule | 0.0 | 0.0 | 4.2 | 0.0 |
| Same hour yesterday | 0.0 | 0.0 | 0.0 | 3.72 |
| Last year's profile | 0.0 | 0.0 | 0.0 | 3.05 |

of the best possible saving, against 94.1% for the residual-load rule.
+105k EUR
a year more than the rule, for 500 MWh a day.
71.5% vs 70.7%
of days the model and the rule pick exactly the cheapest shift. The model's edge is that its wrong days cost less.
Final model refitted on 2019 to 2024 and scored once on 2025. The 2026 reserve (January to October), scored once: 98.7% against 97.8% for the rule, about 49k EUR a year ahead.
Speaker 4 · Evaluation and results
12

### Notes:
The final model was refitted on 2019 to 2024 and scored once on 2025. No other model touched 2025. It captures 96.4% of the possible saving, against 94.1% for the rule: 4.31 million EUR against 4.20 million, so about 105,000 EUR a year more. The lead over the rule was 1.5 points in validation and 2.3 points here, so it held. Both pick the exact cheapest shift about equally often, so the edge comes from the days each one misses: the model's misses cost less. The 2026 reserve agrees, with a smaller lead.

<!-- Slide number: 13 -->
07 · RESULTS
Is the 105k EUR edge more than luck?
We compared the model and the rule on the same 365 days of 2025, then resampled those days 10,000 times to see how much the edge could move.
BOOTSTRAP OF THE MODEL'S YEARLY LEAD OVER THE RULE, THOUSAND EUR · 95% INTERVALS
105k
7-day blocks
8k
223k
30-day blocks
−1.5k
239k
-50
0
50
100
150
200
250
7-day blocks: the lead stays above zero, from about 8k to 223k EUR a year.
30-day blocks, which keep longer runs of similar days together: the low end reaches about −1.5k EUR.
Verdict: the model beats the rule in more than 97% of resamples, but the size of the edge is uncertain.
Resampling runs of consecutive days, because neighbouring days are alike through the weekly cycle and spells of weather.
Speaker 5 · Robustness and recommendation
13

### Notes:
Speaker 5. The question a sceptic asks: could 105,000 EUR be luck in which days 2025 happened to bring? We resample the year's daily lead 10,000 times, in runs of consecutive days because neighbouring days are alike. With 7-day runs the whole 95% interval is above zero. [Click] With 30-day runs the low end just touches zero. So the model is very likely better than the rule, but the edge could be anything from almost nothing to more than twice 105,000 EUR. This only covers luck within 2025; whether another year would favour the model as much rests on two validation years.

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

### Chart: Mean absolute error by price band, EUR/MWh

| Category | MAE |
| --- | --- |
| below 0 (758 h) | 16.5 |
| 0 to 200 (16,545 h) | 15.9 |
| above 200 (241 h) | 89.1 |

Tried: predicting asinh(price) to shrink the spikes. Saving share fell from 95.6% to 93.1%, because shrinking the big prices also shrinks the gaps between hours, and those gaps pick the shift.
The 2025 weekday whose mean price is closest to the year's median day. Its forecast is too high in every hour and misses how deep the solar dip goes, yet its shift is the cheapest one.
Out-of-fold predictions, 2023 and 2024. Hours above 200 EUR/MWh are predicted 61 EUR/MWh too low on average.
Speaker 5 · Robustness and recommendation
14

### Notes:
Left: the day from slide 1, Thursday 6 March 2025, the 2025 weekday with the median price level. The forecast runs 11 to 66 EUR/MWh too high and misses the depth of the solar dip, but the shape is right, so it picks the cheapest shift. That is the whole argument in one picture: poor MAE on this day, perfect decision. Right: ordinary and negative hours are missed by about 16 EUR/MWh, but the 241 hours above 200 by 89. That is the model's weak point. The one fix the diagnostics suggested, predicting asinh of the price, made the decision worse, so the model stays as it is and the spikes stay a known weakness.

<!-- Slide number: 15 -->
WRAP-UP
Schedule by forecast. Most of the money is in choosing at all
LIMITS
4.2M EUR
Forecast timing
Thin evidence
a year from picking the shift at all, even by the simple residual-load rule, against a fixed 06:00 start.
The grid operators publish at 18:00, after the gate. The rule and the model both assume a bought vendor forecast.
Two validation years and one test year, so the lead over the rule is a narrow call.
+105k EUR
Simplified decision
Missing inputs
on top from the XGBoost forecast. The rule needs the same bought vendor forecast, so the model only adds the cost of running it. 2026 agrees, with a smaller lead.
Shifts stay within one day, premiums do not stack, and the plant does not move the price.
No fuel prices or public holidays yet. LEAR was refitted yearly, not daily.
Price memory
Thank you. Questions?
is the feature group that matters. Spikes above 200 EUR/MWh stay hard to forecast.
Speaker 5 · Robustness and recommendation
15

### Notes:
Most of the money comes from choosing the shift at all, not from the model: the rule alone is worth 4.2 million EUR a year, about 40 times the forecast's own gain. Both options need a vendor forecast of load, solar and wind, because the rule ranks the hours by forecast residual load, which comes from that same forecast. So buying it is a cost of choosing the shift at all, not of the model. On top of that forecast, the model adds about 105,000 EUR a year for the cost of running it on public price history. Keep the rule as the fallback on days the model fails. Limitations, ranked: the input forecasts used here arrive after the gate; the lead over the rule is a narrow call; spikes are missed; the decision is simplified. On the price-taker point: buying adds about 0.18 EUR/MWh to the price paid, under 2% of what the forecast saves. Everyone should be ready to take questions on any slide.
