@echo off
echo ============================================================
echo   Loitering Munition Damage Assessment System v2
echo ============================================================
echo.
echo Looking for Python...
echo.
set "PYTHON=python"
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Activate the project virtual environment first.
    pause
    exit /b 1
)
echo Using: %PYTHON%
echo.
echo [1] Start Web UI
echo [2] Run Stage-0 Tests
echo [3] Install Dependencies
echo [4] Run Stage-0 Smoke Dataset
echo [5] Exit
echo.
set /p c="Select: "
if "%c%"=="1" goto web
if "%c%"=="2" goto test
if "%c%"=="3" goto inst
if "%c%"=="4" goto smoke
if "%c%"=="5" exit /b 0
echo Bad choice
pause
goto :eof
:inst
"%PYTHON%" -m pip install -e ".[ui,test]"
pause
goto :eof
:web
"%PYTHON%" -m streamlit run src\loitering_munition_damage_twin\visualization\app.py
pause
goto :eof
:test
"%PYTHON%" -m unittest discover -s tests -v
pause
goto :eof
:smoke
"%PYTHON%" -m loitering_munition_damage_twin.stage0.smoke --rows 24 --mc-replicates 2
pause
goto :eof
