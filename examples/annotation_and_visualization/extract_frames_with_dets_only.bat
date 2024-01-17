@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Input and output folders
SET INPUT=videos
SET OUTPUT=frames

REM The default frame rate is only used when the csvs alongside videos
REM do not contain frame rates, otherwise the CSV frame rate is used.
SET DEFAULT_FRAME_RATE=5

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_video.py" ^
  -i "%INPUT%" -o %OUTPUT% -frate %DEFAULT_FRAME_RATE% ^
  -p "pipelines/filter_tracks_only.pipe" ^
  -auto-detect-gt viame_csv

pause
