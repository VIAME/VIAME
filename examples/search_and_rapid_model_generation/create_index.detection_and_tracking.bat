@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

REM To change this script to process a directory of videos, as opposed to images change "-l ingest_list.txt" to "-d videos" if videos is a directory with videos

viame.exe index add -l ingest_list.txt --method tracking -install "%VIAME_INSTALL%"

pause
