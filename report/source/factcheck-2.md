# Fact-check 2: the edited report (`dba3803-report.md`)

Checked 2026-11-14 after the step 4 edit (base: `dba3803-report-gemini-factfixed.md`) and the new section 6.2. There was no humanizer pass (Martin's call), so this is the last check.

**Numbers.** `check_numbers.py` finds no number outside `facts.md` and the Claude draft. The only items it lists are section headings (6.1 to 6.3), a figure width (85), 4.16 million (4,160,000 rounded) and the within-day shares 0.42, 0.37, 0.19, 0.43 (0.418, 0.367, 0.194, 0.431 rounded, Main [41]). All five tables are unchanged from the fact-fixed copy.

**Claims fixed in this round:**
1. Section 6.2 said the model's errors are "mostly" in price spikes. The total error is mostly ordinary hours (16,545 hours at MAE 16.1 against 241 at 91.1); the spikes are the largest errors, not most of the error. Reworded.
2. Section 6.2 called the dropped neighbour feature "the neighbouring zones' prices"; it is their price gap to Germany on D-1 (Main [101]).
3. Recommendation 2's "119,000 more than a fixed weekly plan" compares with the weekly plan under the rest rule (3.93 million); now said.
4. Section 6.2 first said a model without recent prices pays the premium "at night"; the notebook's reason covers the night and weekend premium and is a likely explanation (Main [125]). Reworded and added to `facts.md`.

**Checked and right:** every claim in the new section 6.2 traces to Main [124], [125], [174] to [188], `docs/selection-rule-change.md`, `docs/rest-check.md` and `week_ahead` sections 5 and 6. No em dashes, no "think", every citation is on its source card.
