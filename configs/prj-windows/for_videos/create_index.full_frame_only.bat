@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=C:\Program Files\VIAME

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

python.exe "%VIAME_INSTALL%\configs\ingest_video.py" --init -l input_list.txt -p pipelines\ingest_list.res.full_frame.pipe --build-index --ball-tree -archive-width 1400 -archive-height 1000 -install "%VIAME_INSTALL%"

pause
