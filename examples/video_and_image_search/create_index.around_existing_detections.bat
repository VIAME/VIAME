@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

viame.exe index add -l ingest_list.txt -id input_detections.csv --method existing -install "%VIAME_INSTALL%"

pause
