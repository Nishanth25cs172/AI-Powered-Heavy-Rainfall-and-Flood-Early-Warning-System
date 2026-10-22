#!/usr/bin/env bash
echo "===================================================================="
echo "Starting AI-Powered Heavy Rainfall and Flood Early Warning System"
echo "===================================================================="

echo "1. Verifying Unit Test Suite..."
python3 -m unittest discover -s tests -p "test_*.py"

echo "2. Launching Flask Backend Server..."
python3 backend/app.py
