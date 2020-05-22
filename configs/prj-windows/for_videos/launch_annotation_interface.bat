@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=videos
SET CACHE_DIRECTORY=database

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

sealtk.exe --pipeline-directory %VIAME_INSTALL%\configs\pipelines\embedded_dual_stream --theme %VIAME_INSTALL%\configs\gui-params\dark_gui_settings.ini
