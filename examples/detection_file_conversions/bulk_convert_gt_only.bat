@echo off

REM Convert every annotation file under a folder into another format using only
REM the annotation files themselves: any imagery next to them is ignored, and the
REM frame names and numbers stored in the annotations carry through.

SET VIAME_INSTALL=.\..\..

SET INPUT_FOLDER=..\object_detector_training\training_data_mouss
SET OUTPUT_FOLDER=example_output
SET OUTPUT_FORMAT=coco

CALL "%VIAME_INSTALL%\setup_viame.bat"

viame.exe convert "%INPUT_FOLDER%" "%OUTPUT_FOLDER%" ^
  -o %OUTPUT_FORMAT% --no-images

PAUSE
