@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Core processing options
SET INPUT=insert_foldername_here
SET OUTPUT=output

REM Optional GPS metadata. Leave FLIGHT_LOG empty to auto-detect an imagelog.json
REM in the image folder or embedded EXIF GPS; otherwise set it to an FMCLOG CSV.
SET FLIGHT_LOG=

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Recommended sequential-registration settings plus --geo-anchor, which fits a
REM GLOBAL GPS-to-pixel transform: it places featureless water frames by dead-
REM reckoning and reports how far the sequential feature chain has drifted.
SET FLIGHT_LOG_ARG=
IF NOT "%FLIGHT_LOG%"=="" SET FLIGHT_LOG_ARG=--flight-log %FLIGHT_LOG%

python.exe "%VIAME_INSTALL%\configs\reconstruct_3d.py" "%INPUT%" --output "%OUTPUT%" --planar --coverage-class suppressed --visualize --affine --consistency-filter --xcam-robust --xcam-low-drift --geo-anchor %FLIGHT_LOG_ARG%

pause
