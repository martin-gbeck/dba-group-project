"""Build DBA3803_Group10_Report_v4.docx from v3: the repo's numbers, the timing fix and
the two redrawn figures, edited run by run so formatting, tables and figures survive.

Run from the project root: `uv run --with python-docx python report/group/source/make_v4.py`.
"""

import shutil
import sys
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

GROUP = Path("report/group")
V3 = GROUP / "DBA3803_Group10_Report_v3.docx"
V4 = GROUP / "DBA3803_Group10_Report_v4.docx"
FIGURES = GROUP / "figures"


def replace_in_paragraph(p: Paragraph, old: str, new: str) -> None:
    """Replace `old` once in the paragraph, inside one run when it fits, else across
    runs, keeping the first touched run's formatting."""
    for run in p.runs:
        if old in run.text:
            assert run.text.count(old) == 1, (old, run.text)
            run.text = run.text.replace(old, new)
            return
    full = "".join(run.text for run in p.runs)
    assert full.count(old) == 1, (old, full)
    start = full.index(old)
    end = start + len(old)
    position, first = 0, True
    for run in p.runs:
        text = run.text
        a, b = position, position + len(text)
        position = b
        if b <= start or a >= end:
            continue
        cut_from, cut_to = max(start, a) - a, min(end, b) - a
        run.text = text[:cut_from] + (new if first else "") + text[cut_to:]
        first = False


def set_cell(table: Table, row: int, col: int, old: str, new: str) -> None:
    cell = table.rows[row].cells[col]
    assert cell.text == old, (row, col, cell.text, old)
    replace_in_paragraph(cell.paragraphs[0], old, new)
    assert cell.text == new


shutil.copyfile(V3, V4)
doc = Document(str(V4))
blocks = list(doc.element.body.iterchildren())


def para(i: int) -> Paragraph:
    return Paragraph(blocks[i], doc)


def table(i: int) -> Table:
    return Table(blocks[i], doc)


# timing of the grid-operator forecasts (only wind and solar come after the gate)
replace_in_paragraph(
    para(71),
    "publish their day-D forecasts at 18:00 on D-1, after the 11:55 deadline.",
    "publish their day-D wind and solar forecasts at 18:00 on D-1, after the 11:55 deadline; the load forecast comes out before it.",
)
replace_in_paragraph(
    para(150), "since the grid operators publish theirs too late.",
    "since the grid operators publish their wind and solar forecasts too late.",
)
replace_in_paragraph(
    para(155), "The grid-operator forecasts arrive after the deadline,",
    "The grid-operator wind and solar forecasts arrive after the deadline,",
)

# section 7, permutation importance (main notebook, section 6)
replace_in_paragraph(para(131), "costs 64 points of saving share", "costs 61 points of saving share")
replace_in_paragraph(
    para(131),
    "(3.1 and 1.9 points), followed by the 30-day mean and yesterday’s maximum,",
    "(2.7 and 1.4 points), followed by yesterday’s maximum and the 30-day mean,",
)

# section 8.2 (week_ahead section 8)
replace_in_paragraph(para(138), " (86.4%) and a ", " (86.9%) and a ")
replace_in_paragraph(para(138), "calendar (84.4%). The gap is inside the fold SD of 2.5 points",
                     "calendar (83.8%). The gap is inside the fold SD of 5.2 points")
t6 = table(140)
set_cell(t6, 3, 1, "89.9%", sys.argv[1])  # weekly, last week, 2025
set_cell(t6, 3, 2, "€4.02M", "€4.00M")
set_cell(t6, 3, 4, "21", "19")
set_cell(t6, 4, 1, "90.1%", sys.argv[2])  # weekly, week-ahead model, 2025
set_cell(t6, 4, 4, "16", "11")
replace_in_paragraph(para(142), "€4.02M a year", "€4.00M a year")
replace_in_paragraph(para(142), "only 21 start changes", "only 19 start changes")
replace_in_paragraph(para(142), "€293,000 a year", "€308,000 a year")
replace_in_paragraph(para(142), "€135,000 to €458,000", "€150,000 to €483,000")
replace_in_paragraph(para(142), "€53,000", "€68,000")
replace_in_paragraph(para(142), "−€151,000 to €216,000", "−€137,000 to €244,000")
replace_in_paragraph(para(142), "(92.9%, €5.32M a year;", f"({sys.argv[3]}, €5.33M a year;")

# Table 7
t7 = table(146)
set_cell(t7, 2, 1, "€11.19M", "€11.21M")
set_cell(t7, 2, 2, "€4.02M", "€4.00M")
set_cell(t7, 2, 3, "26.4%", "26.3%")

# conclusion
replace_in_paragraph(para(161), "€4.02M a year with legal rest", "€4.00M a year with legal rest")

# Table E1
e1 = table(225)
for row, values in enumerate(sys.argv[4].split(";"), start=1):
    for col, value in enumerate(values.split(","), start=1):
        if value:
            old = e1.rows[row].cells[col].text
            if old != value:
                set_cell(e1, row, col, old, value)

# Table E2
e2 = table(228)
set_cell(e2, 3, 1, "92.9%", sys.argv[3])
set_cell(e2, 3, 2, "€5.32M", "€5.33M")
set_cell(e2, 3, 4, "17", "15")
set_cell(e2, 4, 1, "93.0%", sys.argv[5])
set_cell(e2, 4, 4, "15", "10")


# figures 5 and E1: swap the image bytes, keep the size in the page
def replace_image(paragraph_index: int, png: Path) -> None:
    blip = blocks[paragraph_index].xpath(".//a:blip")[0]
    rid = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
    part = doc.part.related_parts[rid]
    part._blob = png.read_bytes()


replace_image(129, FIGURES / "fig5-permutation-importance.png")
replace_image(230, FIGURES / "figE1-schedules-2025.png")
doc.save(str(V4))
print("wrote", V4)
