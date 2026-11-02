@echo off
REM ==============================================================================
REM SolarVision AI -- Automated Setup Script for Windows
REM ==============================================================================
echo ==============================================================================
echo   SolarVision AI: Environment Setup (CPU-Optimized)
echo ==============================================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH. Please install Python 3.11+.
    pause
    exit /b 1
)

echo [1/4] Checking Virtual Environment...
if not exist .venv (
    echo Creating virtual environment in .venv...
    python -m venv .venv
) else (
    echo Virtual environment .venv already exists.
)

echo [2/4] Activating Virtual Environment and Upgrading Pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [3/4] Installing PyTorch CPU Wheels...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo [4/4] Installing Project Dependencies...
pip install -r requirements.txt
pip install -e .

echo ==============================================================================
echo   [SUCCESS] Setup Completed!
echo   To launch the Streamlit Dashboard, double-click run.bat or run:
echo     run.bat
echo ==============================================================================
pause
