@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

SET DIVE_VIAME_INSTALL_PATH=%VIAME_INSTALL%

"%VIAME_INSTALL%\dive\DIVE-Desktop.exe"
