@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=C:\Program Files\VIAME
SET VIAME_RUN_SCRIPTS=%VIAME_INSTALL%\configs

CALL "%VIAME_INSTALL%\setup_viame.bat" >nul

REM Run GUI launcher

echo Script temporarily disabled on Windows

PAUSE
