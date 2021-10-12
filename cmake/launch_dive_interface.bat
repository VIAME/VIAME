@echo off

REM Setup VIAME Paths (set path if script moved to another directory)

SET VIAME_INSTALL=.

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Launch the DIVE GUI

SET DIVE_VIAME_INSTALL_PATH=%VIAME_INSTALL%

"%VIAME_INSTALL%\dive\DIVE-Desktop.exe"

