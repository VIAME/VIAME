@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

pipeline_runner.exe -p "%VIAME_INSTALL%\configs\pipelines\detector_yolo_default.pipe" ^
                    -s input:image_list_file=input_image_list_small_set.txt

pause
