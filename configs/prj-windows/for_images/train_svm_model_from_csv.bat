@echo off

REM Input locations and types

SET INPUT_DIRECTORY=training_data
SET ANNOTATION_TYPE=viame_csv

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=C:\Program Files\VIAME

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

python.exe "%VIAME_INSTALL%\configs\process_video.py" --init -d %INPUT_DIRECTORY% -p pipelines\index_default.svm.pipe -o database --build-index -auto-detect-gt %ANNOTATION_TYPE% -install "%VIAME_INSTALL%"

REM Perform actual SVM model generation

SET SVM_TRAIN_IMPORT=import viame.arrows.smqtk.smqtk_train_svm_models as trainer
SET SVM_TRAIN_START=trainer.generate_svm_models()

python.exe -c "%SVM_TRAIN_IMPORT%;%SVM_TRAIN_START%"

pause

