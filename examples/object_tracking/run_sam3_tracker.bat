@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run SAM3 single-target tracker (requires SAM3 add-on)

viame "%VIAME_INSTALL%\configs\pipelines\utility_track_selections_sam3.pipe" ^
      -s input:video_filename=input_list.txt

PAUSE
