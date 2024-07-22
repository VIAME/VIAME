@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run score tracks on data for singular metrics

python "%VIAME_INSTALL%\configs\score_results.py" ^
 -computed detections.csv -truth groundtruth.csv ^
 -threshold 0.10 -trk-mot-stats stats_nocls.txt

PAUSE
