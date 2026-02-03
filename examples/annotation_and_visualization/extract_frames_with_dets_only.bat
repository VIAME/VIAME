@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Input and output folders
SET INPUT=videos
SET OUTPUT=frames

REM The default frame rate is only used when the csvs alongside videos
REM do not contain frame rates, otherwise the CSV frame rate is used.
SET DEFAULT_FRAME_RATE=5

REM Set to true to output CSVs for the extracted frames (renumbered to match
REM the output frame numbering). The output CSVs will be in the output folder.
SET OUTPUT_CSVS=true

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

IF "%OUTPUT_CSVS%"=="true" (
  SET PIPELINE=pipelines/filter_tracks_only_adjust_csv.pipe
) ELSE (
  SET PIPELINE=pipelines/filter_tracks_only.pipe
)

python.exe "%VIAME_INSTALL%\configs\process_video.py" ^
  -i "%INPUT%" -o %OUTPUT% -frate %DEFAULT_FRAME_RATE% ^
  -p "%PIPELINE%" ^
  -auto-detect-gt viame_csv

pause
