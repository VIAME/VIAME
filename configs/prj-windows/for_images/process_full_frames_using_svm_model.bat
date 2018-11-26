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

pipeline_runner.exe -p "%VIAME_INSTALL%\configs\pipelines\full_frame_classifier_svm.pipe" ^
                    -s input:video_filename=%INPUT_LIST% ^
                    -s input:frame_time=%INPUT_FRAME_RATE% ^
                    -s downsampler:target_frame_rate=%PROCESS_FRAME_RATE%

pause
