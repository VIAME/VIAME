@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

python.exe "%VIAME_INSTALL%\configs\ingest_video.py" --init -l input_list.txt -id input_detections.csv -p pipelines\ingest_list_and_detections.res.pipe --build-index --ball-tree -install "%VIAME_INSTALL%"

pause
