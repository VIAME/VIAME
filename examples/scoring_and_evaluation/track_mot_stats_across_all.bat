@echo off

REM Setup VIAME Paths (no need to run multiple times if you already ran it)

SET VIAME_INSTALL=%~dp0\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Report multi-object tracking statistics, treating all categories as one.
REM
REM MOTA, MOTP, IDF1, identity switches, fragmentation and the HOTA family are
REM all computed in the same pass as the detection metrics.

viame_score_results.exe ^
 --computed detections.csv --truth groundtruth.csv ^
 --iou 0.5 --conf 0.10 ^
 --output-summary output_mot_stats.txt ^
 --output-metrics output_mot_stats.json

pause
