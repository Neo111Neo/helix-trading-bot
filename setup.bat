@echo off
title HELIX setup
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY goto nopython
echo Creating a private Python environment in .venv ...
%PY% -m venv .venv
if errorlevel 1 goto failed
echo Installing packages ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
if not exist .env copy .env.example .env >nul
echo.
echo Setup finished.
echo   start.bat  = live Hyperliquid prices, paper trading
echo   demo.bat   = simulated market, to see the terminal move with no keys
echo Add your keys to the .env file (open it with Notepad) whenever you have them.
echo.
pause
exit /b 0

:nopython
echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" on the first installer screen. Then run setup.bat again.
pause
exit /b 1

:failed
echo.
echo Setup failed. Scroll up for the error message.
pause
exit /b 1
