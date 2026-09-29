@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run pipeline (requires the HabCam add-on)

kwiver.exe runner "%VIAME_INSTALL%\configs\pipelines\detector_habcam_measure_scallops_one_class_metadata.pipe" ^
                  -s input:video_filename=input_image_list_habcam.txt

pause
