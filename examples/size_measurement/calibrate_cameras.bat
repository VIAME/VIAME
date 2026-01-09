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
REM Additional options:
REM   -a              Auto-detect grid size (recommended)
REM   -x GRID_X       Number of inner corners in grid width (default: 6)
REM   -y GRID_Y       Number of inner corners in grid height (default: 5)
REM   -q SQUARE_SIZE  Width of calibration square in mm (default: 80)
REM   -s FRAME_STEP   Process every Nth frame (default: 1)
REM   -g              Show GUI with detection results

python.exe "%VIAME_INSTALL%\tools\calibrate_cameras.py" -a -q 80 -o calibration_matrices.json %*

pause
