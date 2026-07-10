@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Core processing options. INPUT is either a single folder of images or a
REM multi-camera rig folder containing PORT/STAR/CENTER subfolders.
SET INPUT=insert_foldername_here
SET OUTPUT=output

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Sequential registration + prior coverage WITHOUT any GPS metadata.
REM detect_prior_coverage.py chains affine frame-to-frame registrations, uses
REM a robust cluster consensus for the rig-constant cross-camera transform,
REM carries a moving average of chained motion across featureless open-water
REM gaps, and pseudo-georeferences the site from the chains so within-site
REM revisits are still detected. Writes prior_coverage.csv, revisits.csv,
REM coverage_map.png and a thumbnail visualization into %OUTPUT%.
python.exe "%VIAME_INSTALL%\configs\detect_prior_coverage.py" "%INPUT%" --method hybrid --output "%OUTPUT%"

pause
