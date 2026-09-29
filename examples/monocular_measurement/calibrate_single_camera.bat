@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run calibration pipeline
REM
REM Edit the image list and the checkerboard square size below as needed

kwiver.exe runner "%VIAME_INSTALL%\configs\pipelines\utility_calibrate_single_camera.pipe" ^
                  -s input:video_filename=calibration_images.txt ^
                  -s global:square_size=80

pause
