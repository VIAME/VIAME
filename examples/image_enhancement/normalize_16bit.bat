@echo off

REM Setup VIAME Paths (no need to run multiple times if you already ran it)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run pipeline to normalize 16-bit imagery to 8-bit

viame.exe "%VIAME_INSTALL%\configs\pipelines\filter_normalize_16bit.pipe" ^
          -s input:video_filename=input_list_16bit_images.txt
