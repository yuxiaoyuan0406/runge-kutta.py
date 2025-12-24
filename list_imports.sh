#!/usr/bin/env bash
set -euo pipefail

PATTERN='^[[:space:]]*(from[[:space:]]+[^[:space:]]+[[:space:]]+import|import)[[:space:]]+'

find . \
  \( -path './.git' -o -path './.git/*' \
     -o -path './data' -o -path './data/*' \
     -o -path './.vscode' -o -path './.vscode/*' \
     -o -path './.venv' -o -path './.venv/*' \
     -o -path './venv' -o -path './venv/*' \
     -o -path './env' -o -path './env/*' \
     -o -path '*/__pycache__' -o -path '*/__pycache__/*' \
     -o -path '*/.mypy_cache' -o -path '*/.mypy_cache/*' \
     -o -path '*/.pytest_cache' -o -path '*/.pytest_cache/*' \
     -o -path '*/.tox' -o -path '*/.tox/*' \
     -o -path './build' -o -path './build/*' \
     -o -path './dist' -o -path './dist/*' \) -prune -o \
  -type f -name '*.py' -print0 |
  xargs -0 grep -nH --color=never -E "$PATTERN" || true
