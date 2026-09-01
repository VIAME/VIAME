@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Processing options
SET INPUT_FOLDER=..\object_detector_training\training_data_mouss
SET INPUT_FORMAT=viame_csv
SET OUTPUT_FOLDER=example_output
SET OUTPUT_FORMAT=coco_json
SET OUTPUT_EXTENSION=json

REM Setup paths and run command. Unlike bulk_convert_gt_plus_data this does
REM not require the source imagery or videos to be present, and uses the image
REM names or timestamps stored in the input annotation files instead.
SET PIPELINE=pipelines\convert_%INPUT_FORMAT%_to_%OUTPUT_FORMAT%_gt_only.pipe

CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_video.py" ^
  -i "%INPUT_FOLDER%" -o "%OUTPUT_FOLDER%" ^
  -p %PIPELINE% -output-ext %OUTPUT_EXTENSION% ^
  -auto-detect-gt %INPUT_FORMAT% --gt-only --no-reset-prompt

PAUSE
