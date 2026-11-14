"""Merge Gemini's fact-fixed additions into a copy of the fact-fixed report, each piece
where Gemini placed it, without rewording around it.

Run from report/source: `python3 merge.py` -> dba3803-report-merged.md
"""

import re
from pathlib import Path

report = Path("dba3803-report-gemini-factfixed.md").read_text()
pieces = [
    p.strip()
    for p in re.split(r"^=== PLACE:.*===$", Path("gemini-additions-factfixed.md").read_text(), flags=re.M)
    if p.strip()
]
h61, table3, para_t3, sec62, para71, table4, para_t4, table5, para73, in_short, rec2, lim1, lim8, ref = pieces


def paragraph_with(text: str, needle: str) -> str:
    """The one paragraph (blank-line separated block) of the report containing the needle."""
    hits = [b for b in text.split("\n\n") if needle in b]
    assert len(hits) == 1, (needle, len(hits))
    return hits[0]


def table_block(text: str, caption_start: str) -> str:
    """A Markdown table plus its caption line."""
    caption = paragraph_with(text, caption_start)
    start = text.index(caption)
    table_start = text.rfind("\n\n", 0, start - 2) + 2
    return text[table_start : start + len(caption)]


def line_with(text: str, needle: str) -> str:
    """The one line of the report containing the needle (a list item)."""
    hits = [line for line in text.split("\n") if needle in line]
    assert len(hits) == 1, (needle, len(hits))
    return hits[0]


def after(text: str, needle: str, new: str) -> str:
    block = paragraph_with(text, needle)
    return text.replace(block, block + "\n\n" + new, 1)


def replace(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old[:60]
    return text.replace(old, new, 1)


out = report
out = replace(out, table_block(out, "Table 3."), table3)
out = replace(out, table_block(out, "Table 4."), table4)
out = replace(out, table_block(out, "Table 5."), table5)
first = paragraph_with(out, "With all model specifications locked on the validation folds")
out = replace(out, first, h61 + "\n\n" + first)
out = after(out, "provided by the econometric forecast.", para_t3)
out = after(out, "under 0.05% of resamples.", sec62)
out = after(out, "prevailing wholesale price baselines.", para71)
out = after(out, "global dynamic programming loses only 97,000 EUR.", para_t4)
out = after(out, "imposes no economic penalty on the plant.", para73)
out = replace(out, paragraph_with(out, "In short:"), in_short)
out = replace(out, line_with(out, "2. Establish weekly rosters on Sunday"), rec2)
out = replace(out, line_with(out, "1. Forecast publication timeline:"), lim1)
lim7 = line_with(out, "7. Annual recalibration of the LEAR benchmark")
out = replace(out, lim7, lim7 + "\n" + lim8)
out = after(out, "Elmachtoub, A. N. and Grigas", ref)
Path("dba3803-report-merged.md").write_text(out)
