@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..
SET CURRENT_DIR=%CD%

SET DATA_FOLDER="%CURRENT_DIR%\training_data"
SET TRAIN_FOLDER="%CURRENT_DIR%\deep_tracking"
SET GPU_COUNT=1
SET THRESH=0.0

SET SCRIPT_DIR="%VIAME_INSTALL%\lib\python3.6\site-packages\pysot\viame"

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run pipeline

IF NOT EXIST %DATA_FOLDER% (
  ECHO Training Data Folder Does Not Exist
  PAUSE
  EXIT /B
)

IF EXIST %TRAIN_FOLDER% RD /s /q %TRAIN_FOLDER%
MKDIR %TRAIN_FOLDER%

python.exe -m torch.distributed.launch ^
           --nproc_per_node=%GPU_COUNT% ^
           %SCRIPT_DIR%\viame_train_tracker.py ^
           -i %DATA_FOLDER% ^
           -s %TRAIN_FOLDER% ^
           -c %VIAME_INSTALL%\configs\pipelines\models\pysot_training_config.yaml ^
           --threshold %THRESH%

pause
