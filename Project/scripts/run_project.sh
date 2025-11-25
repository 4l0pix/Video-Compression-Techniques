#!/bin/bash

# Change to the script directory
cd "$(dirname "$0")"

# Script names in execution order
SCRIPTS=("run_compressai.py" "ablation.py" "plot_rd_curves.py")

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if all scripts exist
ALL_SCRIPTS_EXIST=true
for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "ERROR: Script $script not found"
        ALL_SCRIPTS_EXIST=false
    fi
done

if [ "$ALL_SCRIPTS_EXIST" = false ]; then
    echo "Required scripts: run_compressai.py, ablation.py, plot_rd_curves.py"
    exit 1
fi

# Execute each script in sequence
for script in "${SCRIPTS[@]}"; do
    echo "Running $script"
    python "$script"

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: $script failed with exit code: $EXIT_CODE"
        if [ -d "venv" ] && [ -n "$VIRTUAL_ENV" ]; then
            deactivate
        fi
        exit $EXIT_CODE
    fi
done

# Deactivate virtual environment if it was activated
if [ -d "venv" ] && [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

echo "All scripts completed successfully"
