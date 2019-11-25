@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..
SET INPUT_IMAGE_PATTERN=/home/matt/Desktop/track_experiment/images/*.png

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

kwiver.exe runner "${VIAME_INSTALL}/configs/pipelines/register_using_homographies.pipe" ^
                  -s input:video_filename=input_list.txt

python.exe create_mosaic.py first_mosaic.png homog.txt %INPUT_IMAGE_PATTERN%
