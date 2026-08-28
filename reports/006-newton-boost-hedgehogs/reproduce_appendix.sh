#!/bin/bash
# APPENDIX-flat-connection: regenerates the appendix record; the script
# asserts every structural claim internally. Separate from reproduce.sh
# so the merged report's path stays untouched (METHOD section 7).
# CPU-only, seconds.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

$PY appendix_flat_connection.py
echo "APPENDIX REPRODUCTION OK"
