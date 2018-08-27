@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=C:\Program Files\VIAME

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

pipeline_runner.exe -p "%VIAME_INSTALL%\configs\pipelines\full_frame_classifier_svm.pipe" ^
                    -s input:image_list_file=input_list.txt

pause
