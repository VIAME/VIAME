@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Processing options
SET INPUT_FOLDER=..\object_detector_training\training_data_mouss
SET INPUT_FORMAT=viame_csv
SET OUTPUT_FOLDER=example_output
SET OUTPUT_FORMAT=coco_json
SET DEFAULT_FRAME_RATE=5

REM Setup paths and run command
SET PIPELINE=pipelines\convert_%INPUT_FORMAT%_to_%OUTPUT_FORMAT%.pipe

CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_video.py" ^
  -i "%INPUT_FOLDER%" -o "%OUTPUT_FOLDER%" -frate %DEFAULT_FRAME_RATE% ^
  -p %PIPELINE% -auto-detect-gt %INPUT_FORMAT% --no-reset-prompt
 
PAUSE
