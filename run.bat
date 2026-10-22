@echo off
echo ====================================================================
echo Starting AI-Powered Heavy Rainfall and Flood Early Warning System
echo ====================================================================

echo 1. Verifying Unit Test Suite...
python -m unittest discover -s tests -p "test_*.py"
if %errorlevel% neq 0 (
    echo [WARNING] Test suite reported failures. Continuing to launch server...
)

echo 2. Launching Flask Backend Server...
python backend/app.py
