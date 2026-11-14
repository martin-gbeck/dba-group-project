# Fact-check 3: the redo through the loop (2026-11-15)

The hand-in version of 2026-11-14 broke the loop's rules (step 4 rewrote about 3,400 words, and Claude wrote section 6.2 and the week-ahead rule text). This redo starts again from `dba3803-report-gemini-factfixed.md` (step 3 of the first round). Every Claude step was done on Opus.

## Step 1 again: Gemini writes what was missing

The missing pieces were listed against the old hand-in version, with bare facts and where each goes, in `gemini-brief-2.txt`: section 6.2 (does the model's behaviour make business sense), the new row and paragraph for Table 3, the week-ahead rule in 7.1, the Table 4 row and paragraph, Table 5's rule rows and the closing paragraphs of 7.3, "In short", recommendation 2, limitations 1 and 8, and the card [11] reference. Gemini got the fact-fixed report, `facts.md` and `sources.txt`, with the same framing as `gemini-brief.txt`, through `agy` (gemini-3.8-flash-high; the `gemini_write` tool was not available in this session). Its reply is `gemini-additions.md`, unchanged.

The first brief gave the facts as near-sentences and Gemini copied them almost word for word, which would have made the text Claude's. That reply was discarded (it is not in the repo), the brief was cut down to bare notes, and Gemini wrote the second reply in its own words.

Gemini also added subsection headings to section 6: 6.1 "Performance on the 2025 test year and 2026 reserve", and 6.2 after the reserve paragraph. Its suggestion of a 6.3 heading came only with the first, discarded reply.

## Step 3 on Gemini's additions (`gemini-additions-factfixed.md`)

Only wrong numbers and wrong or imprecise phrases, in place. `check_numbers.py` finds no number outside `facts.md` and the draft (only the headings 6.1 and 6.2). Fixes:

1. **6.2:** features "selected strictly on shift scheduling performance". MAE breaks ties (rule 1, section 4.3), so this is now "first".
2. **6.2:** on 6 March 2025 "the intraday price profile was captured correctly". The forecast missed how deep the midday dip went (Figure 4). Now "captured well enough to rank the shifts".
3. **6.2:** "Forecast errors concentrate during extreme price spikes" contradicts the rest of the sentence: 241 spike hours against 16,545 ordinary ones, so the spikes are the largest errors, not most of the error (factcheck-2 item 1). Now "are largest during".
4. **7.1:** the rule's premium was "10 x 2/7 EUR/MWh for daytime weekend hours". It is 10 x 2/7 in every daytime hour, the weekend premium spread over the week (Week [40]). Fixed.
5. **7.1:** the model's value comes "when transmission system operator forecasts of tomorrow's load, wind and solar become available". TSO wind and solar forecasts are due only by 18:00, after the auction. "Transmission system operator" was dropped.
6. **7.1:** "a week in advance, those fundamental forecasts do not exist" contradicts the paragraph's own first sentence (vendors sell 10-day forecasts, card [11]). Now "no public wind or solar forecasts exist".
7. **7.1:** the rule "costs nothing to implement or maintain". The fact is that it costs nothing to run. Fixed.
8. **7.3:** the two calls were "locked prior to inspecting 2025 results". They were declared before any results, validation included (Week [56]). Fixed.
9. **7.3:** the model's plan replaces the rule's only if it "decisively" outperforms it. The test is the paired-gap rule, not a decisive win. "Decisively" was removed.
10. **7.3:** anchoring on the rule "maximises annual cost savings". It is the best of the schedules tested, not a maximum. Fixed.
11. **In short:** "the optimal operational policy". Now "the best operational policy we tested".
12. **Recommendation 2:** "strictly satisfies German labour law". Only the 11-hour rest rule (ArbZG §5) was checked. Now "satisfies the statutory 11-hour rest rule".
13. **Limitation 1:** "legally mandated only by 18:00 on D-1, rather than being published at that time, six hours after the day-ahead auction clears at 12:00". The clause read as if the forecasts are not published at 18:00, and the auction closes at 12:00; it clears later. Now "legally due only by 18:00 on D-1, six hours after the day-ahead auction closes at 12:00".
14. **Limitation 1:** the commercial forecast is bought "on the morning of delivery". It has to be bought the morning before, on D-1. Fixed.
15. **Limitation 8:** "This lack of weather fundamentals explains why a rule could match". This is a likely reason, not a shown one (card [11] does not support it). Now "most likely explains".

Change: 34 words out, 50 in, of 3,004.

**Merge (`merge.py`):** each fixed piece was placed where Gemini said in a copy of the fact-fixed report, giving `dba3803-report-merged.md`. Nothing around the pieces was reworded.

## Step 4 (`step4.py`, `dba3803-report-step4.md`)

Only one-for-one plain-word swaps and fixes to bad sentences, all listed in `step4.py`: 97 words out, 78 in, of 8,835 (about 1%).

- **Swaps** (each one everywhere it appears): approximately to about (11), prior to to before (6), procure(s) to buy(s) (5), facility to plant (4), commenced to started, nocturnal to night, expends to spends, exhibit(ed) to show(ed), utilise to use, whilst to while, affords to gives, erroneously to wrongly, fidelity to quality, designated to called, isolated XGBoost to XGBoost alone, forfeits to loses, mandated to due (the 18:00 sentence in section 2), corroborate to confirm, demonstrating to showing, dictate(s) to decide(s), subsequently to then, surpassing to beating, and constitute(s) to is, form or was.
- **Bad sentences:**
  - The spike sentence in 6.2 repeated itself after the step 3 fix and said "error mass". It now says the spike hours do not make up most of the total error.
  - "This performance outperforms" in recommendation 2 is now "This beats".
  - Two figures in the new 7.2 paragraph were each given two or three times over. Each is now given once.
  - Limitation 7's "to match computational baselines across methods" said nothing. It is now "like the other methods".
  - Headings 5.4 and 6.2 asked the same question. 5.4, which is about the errors, is now "Where the model's errors lie".
- **Factcheck-2 claim fixes:** all four concerned text that Gemini has now rewritten (6.2 and recommendation 2). Each was checked again in step 3 (items 1 to 4: spikes, the neighbour gap, the 119,000 figure dropped, the premium reason given as "likely"). The one that also applied to the base text, spikes as "most" of the error in 5.4, is fixed in step 5.
- **18:00:** every mention now says the forecasts are due by 18:00 (sections 2, 6.2, recommendation 4, limitation 1). None says "published at" or "published after".

## Step 5: final check (`dba3803-report-final.md`)

`check_numbers.py` lists only section headings, the Lasso penalty 3.0 and the 1,000,000 (4,055,000 minus 3,055,000). Every claim in the new text was read against `facts.md`, `docs/week-rule.md` and the source cards, and the rest was read again against factcheck-1. Fixes:

1. **5.4:** "Model error concentrates during extreme price spikes". As in factcheck-2 item 1, these are the largest errors, not most of the error. Now "The largest model errors occur during extreme price spikes".
2. **Limitation 3:** "errors reaching 245 EUR/MWh during windless winter spells". The 245 is a daily mean error on one day, 13 December 2024, in one cold spell (Main [180]). Now "daily mean errors of up to 245 EUR/MWh in a windless winter spell".
3. **References:** 7.1 cites Regulation 543/2013 Art. 6(1)(c), but the reference listed only 6(2)(b) and 14(2)(d). 6(1)(c) was added.

Checked and right: all five tables (the new rows against facts sections 9, 15 and 16), the last-week rule numbers (88.1, 87.8, 88.3, 86.1, +5.0, -1.1, SE 3.1, 89.0, 3,979,530, 10/19/20, 101 to 127, 155 to 169, 2 breaks, +22,000, -69,000 and -25,000, 4,001,426 against 4,008,643), the plan figures (+2.0 SE 0.9, +2.7 SE 0.7, +2.2 SE 2.4, +2.2 SE 0.0, 4,129,000 at 94.4%, +128,000, +85,000, +121,000, 4.16 against 4.10 million), 305,000 over last week's prices, and the 6.2 figures (31.4 SE 4.1, 10.3 SE 2.4, 64.4 against 84.0, MAE 19.1 to 17.5 and -0.6 SE 0.3, 91.1 over 241 hours, 16.1 over 16,545 hours, 7.4%). Card [11] is cited only for vendor forecasts reaching ten days ahead. Card [8] is cited for the 18:00 and two-hour deadlines and the week-ahead load forecast. There are no em dashes and no "think".

**Could not verify:** whether a commercial forecast would let a model beat the rule a week ahead. The report says so only as a follow-up. That the missing weather input is why the rule matched the model is the group's own reading, and the text gives it as "most likely".

## Build

`layout.py` carries the hand-in's title block, figure widths and table column widths over to the final text without changing any words. The result is `dba3803-report.md`, built to `report/dba3803-report.docx` (`build_docx.sh`) and `.pdf` (LibreOffice). Before overwriting, the old md was rebuilt and its text matched the old docx, so the docx had no hand edits. The PDF is 21 pages, against 16 for the old hand-in: Gemini's text is longer than Claude's. The outline aimed for 12 to 15 pages, and the brief sets no limit.
