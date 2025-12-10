@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run calibration tool
REM Usage: Provide a video file or image glob pattern as the first argument
REM Example: calibrate_cameras.bat calibration_video.mp4
REM Example: calibrate_cameras.bat "calibration_images\*.png"

python.exe "%VIAME_INSTALL%\tools\calibrate_cameras.py" -j calibration_matrices.json %*

pause
