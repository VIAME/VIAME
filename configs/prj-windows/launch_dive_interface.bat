@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Set current directory for project folder pipe
SET VIAME_PROJECT_DIR=%~dp0

REM Set fixed path to VIAME algorithms for DIVE
SET DIVE_VIAME_INSTALL_PATH=%VIAME_INSTALL%

"%VIAME_INSTALL%\dive\DIVE-Desktop.exe"
