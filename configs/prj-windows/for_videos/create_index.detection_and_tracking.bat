@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=C:\Program Files\VIAME

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

python.exe "%VIAME_INSTALL%\configs\ingest_video.py" --init -d videos -p pipelines\index_default.tut.res.pipe --build-index --ball-tree -install "%VIAME_INSTALL%"

pause
