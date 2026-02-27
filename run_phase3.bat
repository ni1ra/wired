@echo off
echo Configuring MSVC Environment for CUDA...
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo =========================================================
echo INITIATING GESTALT V6: PHASE 3 - SUPER SUPERVISED FINE TUNING
echo [ANTI-CRASH MEASURES ENGAGED]
echo =========================================================
echo.

set GESTALT_BATCH_SIZE=16
set GESTALT_ACCUM_STEPS=2
set GESTALT_SFT_STEPS=100000000
set GESTALT_BRAIN_ONLY=1

:loop
echo.
echo =========================================================
echo [SYSTEM] Starting/Resuming Gestalt Training Loop...
echo [SYSTEM] Target: 100,000,000 Steps (Indefinite Inference)
echo =========================================================
C:\Users\nira\.cargo\bin\cargo run --release --bin gestalt --features cuda -- train --config phase2 --resume

echo.
echo [SYSTEM] Sub-process terminated (Exit Code: %ERRORLEVEL%).
echo [ANTI-CRASH] Rebooting the matrix in 5 seconds to resume from last checkpoint...
set ERRORLEVEL=
timeout /t 5
goto loop
