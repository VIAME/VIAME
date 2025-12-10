@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run calibration tool
REM
REM Usage Mode 1 - Stitched stereo images (left and right horizontally concatenated):
REM   calibrate_cameras.bat calibration_video.mp4
REM   calibrate_cameras.bat "calibration_images\*.png"
REM
REM Usage Mode 2 - Separate left/right images:
REM   calibrate_cameras.bat --left left_video.mp4 --right right_video.mp4
REM   calibrate_cameras.bat --left .\left_images\ --right .\right_images\
REM   calibrate_cameras.bat --left "left\*.png" --right "right\*.png"
REM
REM Additional options: -x GRID_X -y GRID_Y -q SQUARE_SIZE_MM -s FRAME_STEP -g (gui)

python.exe "%VIAME_INSTALL%\tools\calibrate_cameras.py" -j calibration_matrices.json %*

pause
