"""Step 4 of the loop on the merged report: one-for-one plain-word swaps and fixes to bad
sentences only. Every change is listed here, so the edit can be read in full.

Run from report/source: `python3 step4.py` -> dba3803-report-step4.md
"""

from pathlib import Path

text = Path("dba3803-report-merged.md").read_text()

# Plain-word swaps, every occurrence: (formal, plain, expected count).
SWAPS = [
    ("approximately", "about", 11),
    ("prior to", "before", 6),
    ("procures", "buys", 3),
    ("would procure", "would buy", 1),
    ("Procure commercial", "Buy commercial", 1),
    ("commenced", "started", 2),
    ("nocturnal", "night", 2),
    ("expends", "spends", 1),
    ("exhibit wide", "show wide", 1),
    ("exhibited", "showed", 1),
    ("utilise", "use", 1),
    ("whilst", "while", 1),
    ("affords", "gives", 1),
    ("facility", "plant", 4),
    ("erroneously", "wrongly", 1),
    ("fidelity", "quality", 1),
    ("designated the", "called the", 1),
    ("from isolated XGBoost", "from XGBoost alone", 1),
    ("forfeits", "loses", 1),
    ("legally mandated only by 18:00", "legally due only by 18:00", 1),
    ("corroborate that", "confirm that", 1),
    ("corroborating", "confirming", 1),
    ("demonstrating that", "showing that", 1),
    ("dictates", "decides", 2),
    ("forecasts dictate", "forecasts decide", 1),
    ("subsequently", "then", 2),
    ("surpassing", "beating", 1),
    ("constitutes the dominant", "is the main", 1),
    ("constitute an untouched", "form an untouched", 1),
    ("this constituted an", "this was an", 1),
]

# Bad sentences: wrong, unclear or broken. (old, new)
FIXES = [
    (  # redundant after the step 3 fix, and unclear ("error mass")
        "Forecast errors are largest during extreme market price spikes, representing our largest point inaccuracies rather than the majority of total error mass.",
        "Forecast errors are largest during extreme market price spikes, although these hours do not make up most of the total error.",
    ),
    ("This performance outperforms choosing", "This beats choosing"),  # "performance outperforms"
    (  # the same numbers given twice and three times over
        "increasing annual savings by 22,000 EUR (+22,000 EUR, reaching a 91.5% saving share",
        "increasing annual savings by 22,000 EUR (reaching a 91.5% saving share",
    ),
    ("4.00 million EUR (4,001,000 EUR, or specifically 4,001,426 EUR)", "4.00 million EUR (4,001,426 EUR)"),
    ("4.01 million EUR (4,009,000 EUR, or 4,008,643 EUR)", "4.01 million EUR (4,008,643 EUR)"),
    (  # "computational baselines" says nothing; the fact is only that it is refitted like the others
        "recalibrated once a year to match computational baselines across methods",
        "recalibrated once a year, like the other methods",
    ),
    (  # two headings asked the same question; 5.4 is about the errors
        "## 5.4 Does the model make sense?",
        "## 5.4 Where the model's errors lie",
    ),
]

for old, new, count in SWAPS:
    assert text.count(old) == count, (old, text.count(old))
    text = text.replace(old, new)
for old, new in FIXES:
    assert text.count(old) == 1, old
    text = text.replace(old, new)
Path("dba3803-report-step4.md").write_text(text)
