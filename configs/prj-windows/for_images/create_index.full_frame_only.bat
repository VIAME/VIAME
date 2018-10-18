@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=C:\Program Files\VIAME

REM Set these to your maximum image size

SET MAX_IMAGE_WIDTH=1400
SET MAX_IMAGE_HEIGHT=1000

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

python.exe "%VIAME_INSTALL%\configs\process_video.py" --init -l input_list.txt -p pipelines\index_full_frame.res.pipe --build-index --ball-tree -archive-width %MAX_IMAGE_WIDTH% -archive-height %MAX_IMAGE_HEIGHT% -install "%VIAME_INSTALL%"

pause
