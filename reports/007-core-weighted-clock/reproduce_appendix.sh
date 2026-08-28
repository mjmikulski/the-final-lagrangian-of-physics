#!/bin/bash
# APPENDIX-no-connection-from-M: regenerates the appendix results and
# figure; the producer script asserts every structural claim internally.
# Separate from reproduce.sh so the merged report's path stays untouched
# (METHOD section 7). CPU-only, seconds.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

$PY appendix_no_connection.py
$PY make_appendix_figure.py

$PY - <<'EOF'
import os
p = "results/fig_appendix_connections.png"
assert os.path.exists(p) and os.path.getsize(p) > 20000, p
print("appendix figure present:", p)
EOF
echo "APPENDIX REPRODUCTION OK"
