@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

python.exe "%VIAME_INSTALL%\configs\process_video.py" --init -l ingest_list.txt -id input_detections.csv -p pipelines\index_existing.pipe -o database --build-index -install "%VIAME_INSTALL%"

pause
