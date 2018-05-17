@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=C:\Program Files\Video-CAT
SET VIAME_RUN_SCRIPTS=%VIAME_INSTALL%\run_scripts

CALL "%VIAME_INSTALL%\setup_viame.bat" 2> nul

REM Run GUI launcher

python.exe "%VIAME_RUN_SCRIPTS%\ingest_video.py" --init -d input_videos --build-index ^
  -frate 1 -fbatch 5 -fskip 95 ^
  -install "%VIAME_INSTALL%"
pause