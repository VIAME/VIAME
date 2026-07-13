@echo off

REM Setup VIAME Paths (no need to run multiple times if you already ran it)

SET VIAME_INSTALL=%~dp0\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Score detections and tracks against groundtruth, treating all categories as one.
REM
REM Writes a metric summary, a metrics json, and plots (precision-recall curve,
REM detection ROC curve, confusion matrix, and score histograms) into output_metrics.

viame_score_results.exe ^
 --computed detections.csv --truth groundtruth.csv ^
 --iou 0.5 ^
 --output-summary output_metrics_summary.txt ^
 --output-metrics output_metrics.json ^
 --output-plots output_metrics

pause
