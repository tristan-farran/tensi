#!/bin/bash
# usage: scripts/sweep.sh SRC_DIR [OUT]
# Run `tensi peek` and `tensi check` over every .py file under SRC_DIR (4 at a time), write one line
# per file to OUT (path|peek summary|number of shape errors), and print the totals.
set -u
SRC=${1:?usage: sweep.sh SRC_DIR [OUT]}
OUT=${2:-sweep.txt}
export NO_COLOR=1
find "$SRC" -name "*.py" -not -path "*/test*" -not -name "setup.py" -not -name "__init__.py" -size +1k | sort |
  xargs -P3 -I{} bash -c 'p=$(timeout 300 tensi peek "{}" 2>&1 | grep -E "tensors?, [0-9]+ fully named" | tail -1); c=$(timeout 300 tensi check "{}" 2>&1 | grep -c "^  x "); echo "{}|$p|$c"' |
  sort > "$OUT"
python3 - "$OUT" <<'PY'
import re, sys
t = n = f = e = k = 0
for line in open(sys.argv[1]):
    path, peek, errs = line.rstrip("\n").rsplit("|", 2)
    m = re.search(r"(\d+) tensors?, (\d+) fully named", peek)
    if m:
        t += int(m.group(1)); n += int(m.group(2)); f += int(m.group(2)) > 0
    e += int(errs); k += 1
print(f"{k} files: {t} tensors, {n} fully named, {f} files with a named tensor, {e} shape errors")
PY
