@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_LIST=input_list.txt
SET INPUT_FRAME_RATE=1
SET PROCESS_FRAME_RATE=1

REM Note: Frame rates are specified in hertz, aka frames per second. If the
REM input frame rate is 1 and the process frame rate is also 1, then every
REM input image in the list will be processed. If the process frame rate
REM is 2, then every other image will be processed.

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_video.py" --init -l "%INPUT_LIST%" -ifrate %INPUT_FRAME_RATE% -frate %PROCESS_FRAME_RATE% -p pipelines\index_default.res.pipe --build-index --ball-tree -install "%VIAME_INSTALL%"

pause
