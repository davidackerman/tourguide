#!/bin/bash
# Inspect all CSV files in the C. elegans dataset

echo "========================================"
echo "Inspecting all C. elegans CSV files"
echo "========================================"
echo ""

CSV_DIR="/nrs/cellmap/ackermand/cellmap/analysisResults/c-elegans/jrc_c-elegans-op50-1"

for csv in "$CSV_DIR"/*.csv; do
    if [ -f "$csv" ]; then
        echo ""
        echo "###############################################################################"
        echo "# $(basename "$csv")"
        echo "###############################################################################"
        pixi run python inspect_csv.py "$csv" 2>/dev/null | grep -A 100 "DATABASE MAPPING" | head -20
        echo ""
    fi
done

echo ""
echo "========================================"
echo "Summary complete!"
echo "========================================"
