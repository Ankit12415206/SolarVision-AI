@echo off
REM ==============================================================================
REM SolarVision AI -- Dashboard Launcher Script for Windows
REM ==============================================================================
echo ==============================================================================
echo   SolarVision AI: Launching Dashboard...
echo ==============================================================================

if not exist .venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo Starting Streamlit server on http://localhost:8501 ...
streamlit run app/main.py
pause
