#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT/data"

for dataset in PrimeIntellect MATH_train GSM8K_train MATH500 GSM8K MBPP HumanEval; do
    echo "Downloading $dataset..."
    python download_data.py --dataset "$dataset"
done

echo "All datasets downloaded."
