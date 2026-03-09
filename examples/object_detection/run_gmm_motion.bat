@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run GMM motion detector
REM Best for stationary camera scenarios with moving objects.

viame "%VIAME_INSTALL%\configs\pipelines\detector_gmm_motion.pipe" ^
      -s input:video_filename=input_list.txt

PAUSE
