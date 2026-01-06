@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

REM To change this script to process a directory of videos, as opposed to images change "-l ingest_list.txt" to "-d videos" if videos is a directory with videos

python.exe "%VIAME_INSTALL%\configs\process_video.py" --init -l ingest_list.txt -p pipelines\index_default.trk.pipe -o database --build-index -install "%VIAME_INSTALL%"

pause
