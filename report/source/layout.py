"""Carry the hand-in's layout (title block, figure widths, table column widths) over to the
final text, changing no words. Run from report/source: `python3 layout.py`."""

import re
from pathlib import Path

old = Path("dba3803-report.md").read_text()
text = Path("dba3803-report-final.md").read_text()
title, subtitle = text.split("\n\n", 2)[:2]
text = text.replace(title + "\n\n" + subtitle + "\n\n", old[: old.index("# 1.")], 1)
separator = re.compile(r"^\|[-|]+\|$", re.M)
widths = iter(separator.findall(old))
text = separator.sub(lambda _: next(widths), text)
for fig, width in re.findall(r"(\.\./figures/fig\d[\w-]*\.png\))(\{width=\d+%\})", old):
    assert text.count(fig) == 1
    text = text.replace(fig, fig + width)
Path("dba3803-report.md").write_text(text)
