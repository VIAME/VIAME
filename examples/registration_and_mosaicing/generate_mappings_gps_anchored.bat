@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Core processing options
SET INPUT=insert_foldername_here
SET OUTPUT=output

REM Optional GPS metadata: a daily FMCLOG CSV or a directory of them. Leave
REM empty to auto-detect an imagelog.json in the image folder or embedded
REM EXIF GPS.
SET FLIGHT_LOGS=

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Sequential registration + prior coverage WITH GPS anchoring.
REM detect_prior_coverage.py calibrates a metres-to-pixels map from the raw
REM pairwise registrations (bounded by the altitude/focal-length expectation),
REM places featureless open-water frames by GPS dead-reckoning, and tracks
REM all observed ground in a geo-referenced occupancy grid so revisits are
REM detected and confirmed by direct registration. Writes prior_coverage.csv,
REM revisits.csv, coverage_map.png and a thumbnail visualization into %OUTPUT%.
SET FLIGHT_LOGS_ARG=
IF NOT "%FLIGHT_LOGS%"=="" SET FLIGHT_LOGS_ARG=--flight-logs %FLIGHT_LOGS%

python.exe "%VIAME_INSTALL%\configs\detect_prior_coverage.py" "%INPUT%" --method hybrid --output "%OUTPUT%" %FLIGHT_LOGS_ARG%

pause
