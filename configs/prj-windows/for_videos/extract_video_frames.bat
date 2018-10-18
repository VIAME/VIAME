@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=videos
SET FRAME_RATE=5
SET START_TIME=00:00:00.00
SET OUTPUT_DIR=frames

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

for /r %%FILE in (%INPUT_DIRECTORY%\*) do ^
  if not exist "%OUTPUT_DIR%\%FILE%" mkdir "%OUTPUT_DIR%\%FILE%" && ^
  ffmpeg.exe -i "%FILE%" -r %FRAME_RATE% -ss %START_TIME% "%OUTPUT_DIR%\%i%\frame%%06d.png"

pause
