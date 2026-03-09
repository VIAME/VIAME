@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run SAM2 single-target tracker (requires SAM2 add-on)

viame "%VIAME_INSTALL%\configs\pipelines\utility_track_selections_sam2.pipe" ^
      -s input:video_filename=input_list.txt

PAUSE
