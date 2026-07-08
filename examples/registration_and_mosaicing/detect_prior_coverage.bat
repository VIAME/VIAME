@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Core processing options. INPUT is either a single folder of images or a
REM multi-camera rig folder containing PORT/STAR/CENTER subfolders (all
REM cameras are processed together). Multiple site folders can be given to
REM share one coverage grid across them (cross-site / cross-day revisits).
SET INPUT=insert_foldername_here
SET OUTPUT=output

REM Optional GPS metadata: a daily FMCLOG CSV or a directory of them. Leave
REM empty to auto-detect an imagelog.json in the image folder or embedded
REM EXIF GPS; with no metadata at all, coverage still works within-site from
REM the registration chains alone.
SET FLIGHT_LOGS=

REM Water/land classifier used to guide registration over featureless water:
REM   svm  = VIAME sea-lion background classifier (most accurate; requires the
REM          VIAME-SEA-LION model pack, errors out if it is missing)
REM   sift = SIFT keypoint-count heuristic (no models, but textured water can
REM          read as land)
REM   auto = SVM when available, else SIFT (default)
SET WATER_METHOD=auto

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Detect previously-observed image regions with the recommended (hybrid)
REM method: affine registration chains + rig-constant cross-camera consensus
REM for precise recent overlap, and a geo-referenced ground-occupancy grid
REM for revisits (later passes, loop closures, earlier sites/days), with GPS
REM dead-reckoning across featureless open water. Main output is
REM %OUTPUT%\prior_coverage.csv - a standard VIAME detection CSV with one
REM polygon row per previously-seen region for EVERY camera frame, class
REM names prior_coverage_sequential / _cross_camera / _revisit. Also writes
REM revisits.csv, coverage_map.png and prior_coverage_vis.png (thumbnail grid
REM STAR ^| CENTER ^| PORT with the water class labels overlaid).
SET FLIGHT_LOGS_ARG=
IF NOT "%FLIGHT_LOGS%"=="" SET FLIGHT_LOGS_ARG=--flight-logs %FLIGHT_LOGS%

python.exe "%VIAME_INSTALL%\configs\detect_prior_coverage.py" "%INPUT%" --method hybrid --water-method %WATER_METHOD% --output "%OUTPUT%" %FLIGHT_LOGS_ARG%

pause
