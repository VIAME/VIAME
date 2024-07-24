@ECHO OFF

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Generate ROC

python "%VIAME_INSTALL%\configs\score_results.py" ^
 -computed detections.csv -truth groundtruth.csv ^
 -det-roc roc_per_class.png --per-class

PAUSE
