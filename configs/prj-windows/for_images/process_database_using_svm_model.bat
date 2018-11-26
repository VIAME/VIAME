@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

pipeline_runner.exe -p "%VIAME_INSTALL%\configs\pipelines\database_apply_svm_models.pipe" ^
                    -s reader:reader:db:video_name=input_list ^
                    -s descriptors:video_name=input_list

pause
