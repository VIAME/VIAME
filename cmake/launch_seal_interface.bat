@echo off

REM Setup VIAME Paths (set path if script moved to another directory)

SET VIAME_INSTALL=.

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Launch the GUI

sealtk.exe --pipeline-directory "%VIAME_INSTALL%\configs\pipelines\embedded_dual_stream" --theme "%VIAME_INSTALL%\configs\gui-params\seal_color_settings.ini"
