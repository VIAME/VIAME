@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)
SET VIAME_INSTALL=.\..\..

REM Processing options
SET OUTPUT_DIRECTORY=output

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

kwiver.exe runner "%VIAME_INSTALL%\configs\pipelines\filter_stereo_depth_map.pipe" ^
                  -s input1:video_filename=left_images.txt ^
                  -s input2:video_filename=right_images.txt ^
                  -o %OUTPUT_DIRECTORY% --no-reset-prompt

pause
