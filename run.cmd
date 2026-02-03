@echo off
echo Installing dependencies (this may take a while for first time setup)...
py -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing dependencies.
    pause
    exit /b %errorlevel%
)

echo.
echo Ingesting data...
py src/ingest.py
if %errorlevel% neq 0 (
    echo Error in data ingestion.
    pause
    exit /b %errorlevel%
)

echo.
echo Starting Application...
py -m streamlit run src/app.py
pause
