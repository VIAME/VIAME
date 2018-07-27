@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

pipeline_runner.exe -p "%VIAME_INSTALL%\configs\pipelines\database_use_svm_models.pipe" ^
                    -s reader:reader:db:video_name=input_list ^
                    -s descriptors:video_name=input_list

pause
