# Selection rule change: fold SD to paired fold gaps

## What changed

The rule that decides between two options in section 5 used to compare the gap in mean
saving share with the larger of the two options' fold SDs (the SD of each option's 2023 and
2024 scores). It now compares the options year by year:

- `gap = A's score - B's score`, once for 2023 and once for 2024
- `SE = SD of the yearly gaps / sqrt(number of years)`
- A leads if the mean gap is more than one SE in its favour; within one SE is a tie.

Saving share decides first, MAE breaks a tie, and the simpler model wins a tie on both. Rules
2 (best setting within a search) and 3 (a wider search replacing a chosen setting) use the
same test.

The fold SD mostly measured how different 2023 and 2024 are, and that difference moves both
models' scores the same way. The paired gap cancels it. With two years the SE rests on one
degree of freedom, and A leads only if it beats B in both years.

## Chosen model

| | Old rule (fold SD) | New rule (paired gaps) |
|---|---|---|
| Method | XGBoost alone | Lasso-then-XGBoost |
| Settings | depth 3, learning rate 0.05, 200 trees | Lasso penalty 3, depth 6, learning rate 0.03, 500 trees |

Lasso-then-XGBoost wins after the first tuning grid and again after the wider grid, where it
beats XGBoost alone in both years (96.3% against 95.5% in 2023, 97.1% against 96.0% in 2024).
Neither random search replaces either method's wider-grid best.

## Headline numbers

| Number | Old | New |
|---|---|---|
| Validation saving share (mean of 2023, 2024) | 0.956 | 0.967 |
| Validation MAE, EUR/MWh | 17.0 | 17.3 |
| Validation MASE | 0.617 | 0.628 |
| 2025 test saving share | 0.964 | 0.959 |
| 2025 residual-load rule saving share | 0.941 | 0.941 (unchanged) |
| 2025 saving against the standard shift, EUR a year | 4.31 million | 4.29 million |
| 2025 lead over the residual-load rule, EUR a year | 105,000 | 83,000 |
| 2025 test MAE, EUR/MWh | 16.4 | 16.2 |
| 2025 bootstrap 95% interval for the lead, 7-day runs, EUR a year | 8,000 to 223,000 | 3,700 to 181,000 |
| 2025 bootstrap 95% interval, 30-day runs, EUR a year | -1,500 to 239,000 | 550 to 198,000 |
| 2025 resamples where the model does no better than the rule (7-day runs) | 1.6% | 2.0% |
| 2026 reserve saving share (rule: 0.978) | 0.987 | 0.991 |
| 2026 reserve lead over the rule, EUR a year | 49,000 | 73,000 |
| 2026 reserve bootstrap 95% interval, EUR a year | -11,000 to 141,000 | 20,000 to 157,000 |

The new model still beats the residual-load rule on 2025 and on the 2026 reserve. Its 2025
lead is smaller than XGBoost alone's, but it now leads the rule in both validation years
(XGBoost alone tied with the rule at default settings), and on the 2026 reserve its lead is
unlikely to be luck, where XGBoost alone's was not.

One feature decision moved: the neighbour spreads now lose outright (they lower the saving
share in both years) instead of tying; they are dropped either way. Calendar and the level
interaction still tie, and the kept feature groups are unchanged.

## Note for the final report

The presentation showed XGBoost alone, chosen under an earlier rule that judged a lead
against each model's fold SD. That rule was replaced by paired fold gaps, which compare the
two models within each validation year, and under it Lasso-then-XGBoost wins.
