@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Core processing options
SET INPUT=insert_foldername_here
SET OUTPUT=output

REM Optional GPS metadata: a daily FMCLOG CSV or a directory of them (an
REM imagelog.json or EXIF GPS in the input folder is auto-detected if left
REM empty; with no metadata at all, revisits are still detected within-site
REM from the registration chains alone).
SET FLIGHT_LOGS=

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Identify loop-closure / revisit events (the platform leaves a location and
REM later returns to image the same ground). Revisit detection lives in
REM detect_prior_coverage.py: a geo-referenced ground-occupancy grid flags
REM frames that re-cover previously seen ground, and land-to-land events are
REM confirmed by direct image registration. Writes revisits.csv and
REM coverage_map.png into %OUTPUT%. Use "--method metadata" for a fast
REM GPS-only pass, or drop --revisits-only to also get the full per-frame
REM prior-coverage polygons.
SET FLIGHT_LOGS_ARG=
IF NOT "%FLIGHT_LOGS%"=="" SET FLIGHT_LOGS_ARG=--flight-logs %FLIGHT_LOGS%

python.exe "%VIAME_INSTALL%\configs\detect_prior_coverage.py" "%INPUT%" --method hybrid --revisits-only --output "%OUTPUT%" %FLIGHT_LOGS_ARG%

pause
