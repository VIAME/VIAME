@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Core processing options
SET INPUT=insert_foldername_here
SET OUTPUT=output

REM Optional GPS metadata for the metadata method (auto-detects imagelog.json /
REM EXIF GPS if left empty; otherwise set it to an FMCLOG CSV).
SET FLIGHT_LOG=

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Identify loop-closure / revisit events (the platform leaves a location and
REM later returns). --method both runs the GPS-metadata and image-registration
REM detectors and cross-references them.
IF NOT EXIST "%OUTPUT%" mkdir "%OUTPUT%"
SET FLIGHT_LOG_ARG=
IF NOT "%FLIGHT_LOG%"=="" SET FLIGHT_LOG_ARG=--flight-log %FLIGHT_LOG%

python.exe "%VIAME_INSTALL%\configs\detect_site_revisits.py" "%INPUT%" --method both --output "%OUTPUT%\site_revisits.csv" %FLIGHT_LOG_ARG%

pause
