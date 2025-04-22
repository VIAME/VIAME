@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

kwiver.exe runner "%VIAME_INSTALL%\configs\pipelines\tracker_generic_proposals.pipe" ^
                  -s input:video_filename=input_list.txt

PAUSE
