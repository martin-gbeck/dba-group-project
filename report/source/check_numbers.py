"""List every number in a report draft that appears in neither the fact sheet nor the
Claude draft, with the sentence it sits in, for the fact-check.

Run: `python3 report/source/check_numbers.py report/source/gemini-raw.md`.
"""

import re
import sys
from pathlib import Path

SOURCE = Path(__file__).parent
NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")


def numbers_in(text: str) -> set[str]:
    """Every number in the text, commas and % signs dropped, so 4,287,000 and 4287000 match."""
    return {match.replace(",", "").rstrip("%") for match in NUMBER.findall(text)}


known = numbers_in((SOURCE / "facts.md").read_text()) | numbers_in(
    (SOURCE / "draft-claude.md").read_text()
)
report = Path(sys.argv[1]).read_text()
sentences = re.split(r"(?<=[.!?])\s+|\n", report)
for sentence in sentences:
    unknown = sorted(numbers_in(sentence) - known)
    if unknown:
        print(f"{unknown}: {sentence.strip()[:200]}")
