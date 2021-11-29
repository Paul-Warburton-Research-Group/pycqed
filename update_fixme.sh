#!/bin/sh
FIXME_LOG=FIXME_list.txt
SRC_DIR=./pycqed/src
echo "Updating $FIXME_LOG"

lines=$(grep -rni "FIXME" "$SRC_DIR"/*.py)
line_count=$(grep -rni "FIXME" "$SRC_DIR"/*.py | wc -l)

echo "There are $line_count lines indicating a FIX is needed:" > "$FIXME_LOG"
echo "$lines" >> "$FIXME_LOG"
cat $FIXME_LOG
