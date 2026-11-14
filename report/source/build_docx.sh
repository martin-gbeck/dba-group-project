#!/bin/sh
# Build a report docx from its Markdown source, styled like the Citi Bike report.
# Usage (from the project root): sh report/source/build_docx.sh <source.md> <out.docx>
set -eu
reference="$HOME/Documents/Programming/citibike-rebalancing/report/final/citibike-report-editable.docx"
pandoc "$1" --from markdown-implicit_figures --to docx --resource-path=report/source \
  --reference-doc="$reference" -o "$2"
