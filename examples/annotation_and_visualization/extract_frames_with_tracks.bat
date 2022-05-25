@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT=videos
SET OUTPUT=frames
SET FRAME_RATE=5

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_video.py" ^
  -i "%INPUT%" -o %OUTPUT% -frate %FRAME_RATE% ^
  -p "pipelines/filter_tracks_only.pipe" ^
  -auto-detect-gt viame_csv

pause
