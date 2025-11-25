#!/bin/bash

# Video Processing Experiment Runner
# Description: Executes the bonus.py compression analysis script


# Change to the script directory
cd "$(dirname "$0")"

# Check if Python script exists
if [ ! -f "bonus.py" ]; then
    echo "ERROR: bonus.py not found in current directory!"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "WARNING: No virtual environment found. Using system Python."
fi

# Print Python version
echo "Python version: $(python --version)"

# Check if CUDA is available in Python
echo "Checking CUDA availability..."
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

echo "Starting the experiment..."
echo "=========================================="

# Run the Python script
python bonus.py

# Capture the exit code
EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Experiment completed successfully!"
else
    echo "Experiment failed with exit code: $EXIT_CODE"
fi

# Deactivate virtual environment if it was activated
if [ -d "venv" ] && [ -n "$VIRTUAL_ENV" ]; then
    deactivate
    echo "Virtual environment deactivated."
fi

echo "=========================================="
