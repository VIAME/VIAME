@echo off

REM Query the indexed database using tracks from a CSV file
REM
REM This script takes an input track CSV (containing bounding boxes to query)
REM and an image list, queries the database for similar descriptors,
REM and outputs results as a track CSV with nearest neighbor scores as confidence.
REM
REM Usage: perform_cli_query.bat [input_tracks.csv] [query_list.txt] [output.csv]
REM
REM Before running, ensure you have:
REM   1. Built an index using one of the create_index.*.bat scripts
REM   2. Created an input track CSV (e.g., query_box.csv)
REM   3. Created an input image list containing the query image paths
REM
REM The default files are configured for the mouss example imagery set.

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Default input files
SET INPUT_TRACKS=%1
SET INPUT_LIST=%2
SET OUTPUT_FILE=%3

IF "%INPUT_TRACKS%"=="" SET INPUT_TRACKS=query_box.csv
IF "%INPUT_LIST%"=="" SET INPUT_LIST=query_list.txt
IF "%OUTPUT_FILE%"=="" SET OUTPUT_FILE=query_results.csv

REM Check if input files exist
IF NOT EXIST "%INPUT_TRACKS%" (
    echo Error: Input track file not found: %INPUT_TRACKS%
    echo Usage: %0 [input_tracks.csv] [query_list.txt] [output.csv]
    pause
    exit /b 1
)

IF NOT EXIST "%INPUT_LIST%" (
    echo Error: Input list file not found: %INPUT_LIST%
    echo Usage: %0 [input_tracks.csv] [query_list.txt] [output.csv]
    pause
    exit /b 1
)

IF NOT EXIST "database" (
    echo Error: Database directory not found. Please run one of the create_index.*.bat scripts first.
    pause
    exit /b 1
)

echo.
echo ============================================
echo Perform CLI Query
echo ============================================
echo.
echo Input tracks:  %INPUT_TRACKS%
echo Input images:  %INPUT_LIST%
echo Output file:   %OUTPUT_FILE%
echo.

REM Start the database if not running
python.exe "%VIAME_INSTALL%\configs\database_tool.py" start 2>nul

REM Run the query pipeline
kwiver.exe runner "%VIAME_INSTALL%\configs\pipelines\query_from_track.pipe" ^
  -s input:video_filename=%INPUT_LIST% ^
  -s track_reader:file_name=%INPUT_TRACKS% ^
  -s track_writer:file_name=%OUTPUT_FILE%

echo.
echo Query complete. Results written to: %OUTPUT_FILE%
echo.

pause
