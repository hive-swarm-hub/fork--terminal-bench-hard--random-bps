#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/agent:${PYTHONPATH:-}"

python3 "$SCRIPT_DIR/run_eval.py" --concurrency 17
