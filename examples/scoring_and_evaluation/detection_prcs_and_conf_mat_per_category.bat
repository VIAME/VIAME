@echo off

REM Setup VIAME Paths (no need to run multiple times if you already ran it)

SET VIAME_INSTALL=%~dp0\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Write precision-recall, ROC and confusion-matrix data, reporting each
REM category separately in addition to the aggregate.
REM
REM The curves are emitted as CSV alongside rendered plots, so they can be
REM replotted or diffed without rerunning the scoring.

viame_score_results.exe ^
 --computed detections.csv --truth groundtruth.csv ^
 --iou 0.5 --per-class ^
 --output-pr-csv output_prc_and_conf_mat_per_class/pr_curve.csv ^
 --output-roc-csv output_prc_and_conf_mat_per_class/roc_curve.csv ^
 --output-conf-csv output_prc_and_conf_mat_per_class/confusion_matrix.csv ^
 --output-plots output_prc_and_conf_mat_per_class

pause
