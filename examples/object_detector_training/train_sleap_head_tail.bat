@echo off
set "VIAME_INSTALL=%~dp0..\.."
call "%VIAME_INSTALL%\setup_viame.bat"
set "TRAIN_DATA=%~1"
if "%TRAIN_DATA%"=="" set "TRAIN_DATA=training_data"
viame train -i "%TRAIN_DATA%" ^
  -c "%VIAME_INSTALL%\configs\pipelines\train_reclassifier_sleap_head_tail.conf" ^
  --threshold 0.0
