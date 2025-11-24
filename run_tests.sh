#!/bin/bash
# This script runs pytest with verbose output and captures both the
# test report (stdout) and application logs (stderr) into a single log file.

# Ensure the tmp directory exists
mkdir -p tmp

# Run tests and redirect all output to the log file
uv pip install -e . --force-reinstall
uv run pytest -s -v &> tmp/pytest.txt
