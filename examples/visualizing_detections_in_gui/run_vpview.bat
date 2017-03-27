@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=%~dp0\..\..\
set VIDTK_MODULE_PATH=%VIAME_INSTALL%\lib\modules

REM Run Pipeline

%VIAME_INSTALL%\bin\vpView.exe

