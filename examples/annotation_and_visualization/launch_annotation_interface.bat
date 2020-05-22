@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

sealtk.exe --pipeline-directory %VIAME_INSTALL%\configs\pipelines\embedded_dual_stream --theme %VIAME_INSTALL%\configs\gui-params\dark_gui_settings.ini
