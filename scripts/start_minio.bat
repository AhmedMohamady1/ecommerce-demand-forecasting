@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM start_minio.bat  —  Start MinIO natively (Windows)
REM
REM Prerequisites:
REM   • minio.exe on PATH  (see plan.md Section 4 — Path A)
REM   • mc.exe    on PATH  (see plan.md Section 4 — Path A)
REM   • .env file in project root (one key=value pair per line, no spaces)
REM
REM Usage:
REM   scripts\start_minio.bat
REM ─────────────────────────────────────────────────────────────────────────────

REM Load .env key=value pairs into environment variables
for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
    set "%%A=%%B"
)

if "%MINIO_ACCESS_KEY%"=="" (
    echo ERROR: MINIO_ACCESS_KEY not found in .env
    exit /b 1
)

set MINIO_ROOT_USER=%MINIO_ACCESS_KEY%
set MINIO_ROOT_PASSWORD=%MINIO_SECRET_KEY%

set MINIO_DATA_DIR=%USERPROFILE%\minio-data
if not exist "%MINIO_DATA_DIR%" mkdir "%MINIO_DATA_DIR%"

echo ^> Starting MinIO server (data dir: %MINIO_DATA_DIR%) ...
start "MinIO Server" minio.exe server "%MINIO_DATA_DIR%" --address :9000 --console-address :9001

echo   Waiting for MinIO to start...
timeout /t 5 /nobreak > nul

echo ^> Configuring mc alias 'local'...
mc.exe alias set local http://localhost:9000 %MINIO_ACCESS_KEY% %MINIO_SECRET_KEY% --api S3v4

echo ^> Creating bucket 'ecommerce-lake' (if not exists)...
mc.exe mb --ignore-existing local/ecommerce-lake

echo.
echo [OK] MinIO is ready!
echo      API endpoint : http://localhost:9000
echo      Web console  : http://localhost:9001
echo      Bucket       : ecommerce-lake
