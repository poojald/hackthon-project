@echo off
echo ========================================
echo Diabetes Detection System
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
echo.

REM Install/Update dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo.

REM Run the application
echo Starting Flask application...
echo.
echo Application will be available at: http://localhost:5000
echo Admin credentials: admin / admin123
echo.
echo Press Ctrl+C to stop the server
echo.
python run.py

pause
