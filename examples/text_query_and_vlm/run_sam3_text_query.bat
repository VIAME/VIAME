@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline (requires the SAM3 add-on)

kwiver.exe runner "%VIAME_INSTALL%\configs\pipelines\tracker_sam3_animals.pipe" ^
                  -s input:video_filename=input_list.txt ^
                  -s tracker:refiner:sam3:text_query="fish"

pause
