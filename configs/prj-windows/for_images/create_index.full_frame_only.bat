@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_LIST=input_list.txt

REM Set these to your maximum image size
SET MAX_IMAGE_WIDTH=1400
SET MAX_IMAGE_HEIGHT=1000

REM Note: Frame rates are specified in hertz, aka frames per second. If the
REM input frame rate is 1 and the process frame rate is also 1, then every
REM input image in the list will be processed. If the process frame rate
REM is 2, then every other image will be processed.
SET INPUT_FRAME_RATE=1
SET PROCESS_FRAME_RATE=1

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_video.py" --init -l "%INPUT_LIST%" -ifrate %INPUT_FRAME_RATE% -frate %PROCESS_FRAME_RATE% -p pipelines\index_full_frame.res.pipe --build-index --ball-tree -archive-width %MAX_IMAGE_WIDTH% -archive-height %MAX_IMAGE_HEIGHT% -install "%VIAME_INSTALL%"

pause
