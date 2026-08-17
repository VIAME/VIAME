@echo off

REM Setup VIAME Paths (no need to run multiple times if you already ran it)

SET VIAME_INSTALL=%~dp0\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Score detections and tracks against groundtruth, reporting each category
REM separately in addition to the aggregate scores.
REM
REM The per-class table reports TP, FP, FN, precision, recall, F1 and average
REM precision for every category, and the summary reports their mean AP.

viame_score_results.exe ^
 --computed detections.csv --truth groundtruth.csv ^
 --iou 0.5 --per-class ^
 --output-summary output_metrics_per_category_summary.txt ^
 --output-metrics output_metrics_per_category.json ^
 --output-plots output_metrics_per_category

pause
