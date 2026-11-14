=== PLACE: new subsection heading before the paragraph starting 'With all model specifications locked on the validation folds...' in Section 6 ===

## 6.1 Performance on the 2025 test year and 2026 reserve

=== PLACE: replaces Table 3 ===

| Method, 2025 | Saving share | Saving, EUR a year | Cheapest shift picked | MAE |
|---|---|---|---|---|
| Final model | 95.9% | 4,287,000 | 73.7% of days | 16.2 |
| Residual-load rule | 94.1% | 4,205,000 | 70.7% | |
| Last week's mean price at each hour | 89.1% | 3,982,000 | 52.0% | 26.8 |
| Same hour yesterday | 83.2% | 3,717,000 | 49.9% | 26.0 |
| Last year's profile | 68.3% | 3,055,000 | 32.9% | |
| Cheapest shift in hindsight | 100% | 4,469,000 | 100% | |

Table 3. The test year, 2025, for 500 MWh a day.

=== PLACE: new paragraph after the paragraph ending '...provided by the econometric forecast.' ===

A simpler day-ahead pricing benchmark is to schedule shifts using each clock hour's mean clearing price over the preceding seven days. In 2025, this historical average achieves an 89.1% saving share (3.98 million EUR per year, or 3,982,000 EUR). Our final predictive model delivers approximately 305,000 EUR per year in additional savings over this benchmark. The predictive model generates this premium because it incorporates transmission system operator forecasts of tomorrow's load, wind, and solar generation, whereas trailing historical prices contain no information regarding tomorrow's weather conditions.

=== PLACE: new subsection at the end of Section 6, after the paragraph ending '...under 0.05% of resamples.' ===

## 6.2 Does the model's behaviour make business sense?

The model's operational behaviour reflects the physical and economic structure of the German wholesale power market. System residual load dictates how far demand reaches along the merit order, where available power generation plants are ranked from cheapest to dearest marginal operating cost. The marginal facility dispatched at that intersection establishes the market clearing price for each hour, determining the diurnal price shape. Because the variable generation cost of the marginal thermal plant depends predominantly on the price of natural gas, and our dataset excludes fuel commodity indices, recent electricity prices serve as an indispensable proxy for the prevailing fuel cost baseline. Feature ablation confirms this economic logic. Omitting price memory causes saving share to collapse by 31.4 percentage points (SE 4.1). Introducing transmission system operator load and renewable forecasts contributes an additional 10.3 percentage points (SE 2.4) once price memory is established. Without price memory, combining calendar indicators with fundamental forecasts achieves a saving share of only 64.4%, falling well behind calendar indicators alone (84.0%). The likely, though unproven, reason is that without an anchored price baseline, the model misinterprets minor dips in residual load as substantial monetary gains, erroneously committing the plant to pay the 10 EUR/MWh unsocial hours staff premium for nocturnal or weekend shifts.

Consistent with our decision-focused evaluation framework, model features were selected strictly on shift scheduling performance rather than point-forecast error. For instance, incorporating price spreads from 11 neighbouring bidding zones reduced point-forecast MAE from 19.1 to 17.5 EUR/MWh, yet degraded shift selection across both validation years (-0.6 percentage points, SE 0.3), leading us to discard them. Shift optimisation depends exclusively on the relative intraday price profile rather than the absolute price level. As illustrated by the representative operations of 6 March 2025 in Figure 4, the model selected the hindsight-optimal shift despite substantial point-forecast bias across all hours, because the intraday price profile was captured correctly.

Forecast errors concentrate during extreme market price spikes, representing our largest point inaccuracies rather than the majority of total error mass. Across validation hours clearing above 200 EUR/MWh (241 hours), the model recorded an MAE of 91.1 EUR/MWh, compared to an MAE of 16.1 EUR/MWh across normal hours between 0 and 200 EUR/MWh (16,545 hours). As detailed in Section 5.4, an inverse hyperbolic sine transformation, asinh(price), was evaluated to compress these extremes, but it impaired shift selection by dampening the hourly spreads needed to rank candidate shifts. We therefore retained untransformed prices and classified price spikes as an operational limitation.

Following our initial presentation, we revised three core aspects of our methodology where the original operational logic proved unsound:

First, we corrected the model selection protocol. The original rule compared mean margins against cross-year standard deviations, which predominantly captured macroeconomic divergence between 2023 and 2024 rather than algorithmic merit. Replacing this with paired annual differences (Section 4.3) changed our final specification from isolated XGBoost to Lasso-then-XGBoost.

Second, unconstrained daily optimisation violated the statutory 11-hour rest rule on 7.4% of days in 2025. Section 7.2 explicitly incorporates and prices this legal constraint.

Third, scheduling daily shifts at 11:55 affords the operating crew merely one day of notice. Section 7.1 resolves this by establishing a weekly roster on Sunday using a simple heuristic that matches model performance, while Section 7.3 allows the day-ahead forecast to adjust planned start times by up to one hour.

Finally, an inherent data constraint remains that data preprocessing cannot resolve: public wind and solar forecasts are legally due only by 18:00 on D-1, well after auction closure. Our operational findings necessarily assume the facility procures morning commercial forecasts of comparable fidelity, as outlined in Section 2.

=== PLACE: new paragraphs at the end of Section 7.1, after the paragraph ending '...wholesale price baselines.' ===

Although commercial vendors offer multi-day wind and solar power forecasts for Germany extending up to ten days ahead (energy & meteo systems n.d.), public transmission system operators do not issue week-ahead renewable generation forecasts. Because our dataset contains no multi-day weather projections, the week-ahead predictive model operates without meteorological inputs and consequently observes little beyond the preceding week's settled prices.

We therefore evaluated a transparent heuristic alternative against the week-ahead model, designated the 'last-week rule'. At 11:55 on each Sunday, this rule calculates each clock hour's mean price over the seven days up to and including Sunday, adds the staff premium averaged across the week (10 EUR/MWh between 22:00 and 06:00, and 10 x 2/7 EUR/MWh for daytime weekend hours), selects the cheapest 8-hour consecutive shift on that resulting profile, and assigns that identical start time from Monday through Sunday. Across the validation folds, the last-week rule achieved a mean saving share of 88.1% (87.8% in 2023 and 88.3% in 2024), whereas the week-ahead model attained 86.1% (82.8% in 2023 and 89.4% in 2024). The paired annual margin of the rule over the model (+5.0 percentage points in 2023 and -1.1 in 2024, SE 3.1) represents an empirical tie under our model selection protocol (Section 4.3). Because the heuristic rule generates no point forecasts and therefore has no MAE, the simpler method wins, and the rule is selected to establish the weekly baseline roster. We also examined eight rule variants across validation, including four-week historical lookbacks and separate weekday and weekend profiles, but none improved upon the baseline rule; accordingly, only the baseline last-week rule was evaluated on the 2025 test dataset.

On the 2025 test year, the last-week rule attained an 89.0% saving share (3.98 million EUR per year, or 3,979,530 EUR), compared to 90.7% (4.05 million EUR, or 4,055,000 EUR) for the week-ahead econometric model. A single test year cannot overturn a two-year validation decision, particularly as the rule's 5.0 percentage point lead in 2023 exceeds the model's 1.7 percentage point margin in 2025. Furthermore, committing a uniform shift time for the entire week provides substantial operational stability: the last-week rule alters start times only 10, 19, and 20 times per year across 2023, 2024, and 2025 respectively, compared to 101 to 127 start changes under the week-ahead model and 155 to 169 changes under daily selection. Because the rule maintains a single start time from Monday through Sunday, the statutory rest requirements evaluated in Section 7.2 are guaranteed to hold within every weekly roster.

The predictive model's true economic value emerges on the day before delivery, when transmission system operator forecasts of tomorrow's load, wind, and solar generation become available; a week in advance, those fundamental forecasts do not exist. This reveals a broader operational principle: not every component of an industrial decision problem requires a machine-learning model. Where a simple heuristic matches algorithmic performance, it offers substantial business advantages: it is straightforward to communicate to workforce staff and costs nothing to implement or maintain.

=== PLACE: replaces Table 4 ===

| Schedule, 2025 | Days that break the rule | Saving share, free choice | Saving share, rest kept | Change in saving, EUR a year |
|---|---|---|---|---|
| Final model (day-ahead) | 27 (7.4%) | 95.9% | 91.7% | -279,000 |
| Week-ahead schedule | 12 (3.3%) | 90.7% | 89.8% | -129,000 |
| Last-week rule, one start a week | 2 (0.5%) | 89.0% | 91.5% | +22,000 |
| Residual-load rule | 43 (11.8%) | 94.1% | 85.8% | -453,000 |
| Cheapest in hindsight | 26 (7.1%) | 100% | 100% | -97,000 |

Table 4. Rest between shifts, 2025.

=== PLACE: new paragraph after the paragraph ending '...global dynamic programming loses only 97,000 EUR.' ===

Because the last-week rule maintains an identical start time throughout each calendar week, it can violate statutory rest requirements only on Monday transitions, occurring when the incoming week's start time is scheduled more than 5 hours earlier than the previous week's operating shift. In 2025, such conflicts arose on only 2 days (0.5% of the year). Enforcing compliance by restricting the incoming week's start time to begin at most 5 hours earlier imposed zero financial penalty in 2025, increasing annual savings by 22,000 EUR (+22,000 EUR, reaching a 91.5% saving share against rest-constrained hindsight). On validation folds, enforcing rest altered the rule's annual savings by -69,000 EUR in 2023 and -25,000 EUR in 2024. Under mandatory statutory rest in 2025, the last-week rule delivers 4.00 million EUR (4,001,000 EUR, or specifically 4,001,426 EUR), effectively matching the 4.01 million EUR (4,009,000 EUR, or 4,008,643 EUR) captured by the day-ahead forecasting model, whilst providing workforce personnel with a full week of advance roster certainty rather than a single day.

=== PLACE: replaces Table 5 ===

| Plan and reach | Saving share, free | EUR a year, free | Saving share, rest kept | EUR a year, rest kept | Days moved |
|---|---|---|---|---|---|
| Model's plan, no move | 90.7% | 4,055,000 | 89.8% | 3,926,000 | 0% |
| Model's plan, 1 hour either way | 93.1% | 4,160,000 | 92.5% | 4,045,000 | 35% |
| Model's plan, 2 hours either way | 93.3% | 4,172,000 | 92.8% | 4,057,000 | 35% |
| Model's plan, 4 hours either way | 93.8% | 4,191,000 | 93.3% | 4,078,000 | 36% |
| Model's plan, 1 hour earlier only | 91.8% | 4,104,000 | 91.1% | 3,984,000 | 13% |
| Model's plan, 4 hours earlier only | 92.3% | 4,125,000 | 91.8% | 4,012,000 | 13% |
| Last-week rule's plan, no move | 89.0% | 3,980,000 | 91.5% | 4,001,000 | 0% |
| Last-week rule's plan, 1 hour either way | 91.7% | 4,100,000 | 94.4% | 4,129,000 | 35% |
| Any start (day-ahead choice) | 95.9% | 4,287,000 | 91.7% | 4,009,000 | |

Table 5. Plan on Sunday, adjust at 11:55 the day before, 2025. Days moved: share of days whose final start differs from the plan, free choice. Model's plan = week-ahead model, rule's plan = last-week rule of 7.1.

=== PLACE: new paragraphs at the end of Section 7.3, after the paragraph ending '...imposes no economic penalty on the plant.' ===

We applied the identical hybrid scheduling framework using the last-week rule of Section 7.1 to establish the baseline Sunday plan, evaluating adjustment reaches of 0 and 1 hour either way. Following our evaluation protocol, two methodological decisions were locked prior to inspecting 2025 results, governed by paired validation differences. First, daily adjustment flexibility is retained only if adjusting the rule's plan by up to one hour outperforms the unadjusted rule plan. On validation folds, permitting a 1-hour adjustment yielded an improvement of +2.0 percentage points under unconstrained scheduling (SE 0.9) and +2.7 percentage points with statutory rest enforced (SE 0.7), leading in both validation years in each case; the daily adjustment mechanism was therefore retained. Second, the week-ahead model's plan supersedes the last-week rule's plan only if it decisively outperforms it on validation. Under unconstrained scheduling, the rule's plan exceeded the model's plan by +2.2 percentage points, but with SE 2.4 this constituted an empirical tie, selecting the simpler rule. With statutory rest enforced, the rule's plan outperformed the model's plan by +2.2 percentage points (SE 0.0), leading across both validation years; the last-week rule's plan was consequently locked as our primary operational baseline.

Our recommended operational policy is therefore to fix the weekly shift on Sunday via the last-week rule and permit the day-ahead forecasting model at 11:55 on D-1 to adjust the daily start time by up to one hour either way. Under mandatory statutory rest in 2025, this hybrid schedule captures 4.13 million EUR per year (specifically 4,129,000 EUR, or 94.4% of the rest-compliant hindsight optimum). This represents an incremental gain of 128,000 EUR over the static rule plan alone (4,001,000 EUR), 85,000 EUR over the week-ahead model's plan with the identical 1-hour adjustment (4,045,000 EUR, or 4.04 million EUR), and 121,000 EUR over selecting shifts freely each day the day before under the same statutory rest constraint (4,009,000 EUR, or 4.01 million EUR).

Under unconstrained scheduling in 2025, combining the week-ahead model's plan with a 1-hour daily adjustment attained 4.16 million EUR (4,160,000 EUR) compared to 4.10 million EUR (4,100,000 EUR) for the rule's plan. However, unconstrained scheduling repeatedly breaches statutory rest mandates, and the two baseline planning methods tied on validation. When operating under statutory rest requirements, anchoring shift schedules to the last-week rule and fine-tuning daily start times with the day-ahead forecast maximises annual cost savings while preserving workforce stability.

=== PLACE: replaces the paragraph starting 'In short:' at the end of Section 1 ===

In short: our final model, a two-stage Lasso linear regression followed by an XGBoost gradient boosted tree on calendar indicators, transmission system operator forecasts, and recent price lags, selects operating shifts that capture 95.9% of the theoretical maximum cost saving in 2025. This performance delivers 4.29 million EUR per year in cost reductions (electricity plus staff premium) against the standard early shift, providing an incremental gain of 83,000 EUR over the physical residual-load rule. This performance advantage held across both validation years and throughout an untouched reserve dataset covering the first nine months of 2026. Because transmission system operators publish no week-ahead wind or solar forecasts, a simple historical rule committing the previous week's cheapest 8-hour shift from Monday through Sunday establishes the weekly roster as effectively as a machine-learning model. The predictive model's primary value arises on the day before delivery, when renewable and load forecasts become available. When statutory rest constraints are enforced, the optimal operational policy combines both approaches: establish a uniform weekly shift on Sunday using the historical rule and allow the 11:55 day-ahead forecast to adjust the daily start time by up to one hour either way. That hybrid schedule captures 4.13 million EUR per year in 2025, surpassing the 4.01 million EUR achieved by choosing each day freely the day before under the identical legal rest constraint.

=== PLACE: replaces Recommendation 2 in Section 8 ===

2. Establish weekly rosters on Sunday using the last-week rule and allow the 11:55 day-ahead forecast to adjust start times by up to one hour either way, preserving 11 hours of statutory rest. In the absence of public week-ahead wind and solar forecasts, the simple historical rule matches an econometric model while maintaining a single uniform start time across the week. This operating policy provides factory personnel with one week of advance roster notice, strictly satisfies German labour law, and generated 4.13 million EUR in savings in 2025. This performance outperforms choosing shifts freely each day under the identical rest constraint (4.01 million EUR) and delivers an incremental 128,000 EUR per year over the unadjusted weekly rule plan alone. The predictive model's operational role is strictly focused on the day before delivery, where fundamental forecasts are available to fine-tune shift timing.

=== PLACE: replaces Limitation 1 in Section 9 ===

1. Forecast publication timeline: Transmission system operator wind and solar generation forecasts in our dataset are legally mandated only by 18:00 on D-1, rather than being published at that time, six hours after the day-ahead auction clears at 12:00 (Commission Regulation (EU) No 543/2013, Art. 14(2)(d)). Only the total load forecast is required to be published prior to auction gate closure (Art. 6(2)(b)). Consequently, all empirical findings in this study, including those of the physical residual-load heuristic, assume the plant procures commercial forecasts of comparable quality on the morning of delivery before 12:00. A valuable follow-up study should rerun and validate the models using an archived commercial vendor forecast feed.

=== PLACE: new limitation 8 in Section 9, after Limitation 7 ===

8. Absence of week-ahead renewable forecasts: The week-ahead forecasting model operated without meteorological inputs because public transmission system operators do not issue wind and solar generation projections at a multi-day horizon. This lack of weather fundamentals explains why a simple historical rule was able to match the predictive model's performance on a weekly horizon. Commercial vendors offer wind and solar power forecasts for European markets, including Germany, with horizons extending up to ten days (energy & meteo systems n.d.). A worthwhile follow-up study would retest the week-ahead predictive model incorporating such commercial forecasts, which represent the exogenous input most likely to allow an econometric model to outperform the historical rule a week in advance.

=== PLACE: in References, inserted in alphabetical order between Elmachtoub and Grigas (2022) and Fraunhofer ISE (n.d.) ===

energy & meteo systems (n.d.) Market forecasts for wind and solar power in Europe. Available at: https://www.emsys-renewables.com/products/power_forecasts/market-forecasts-wind-solar.php
