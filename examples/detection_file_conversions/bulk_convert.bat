@echo off

REM Convert every annotation file under a folder into another format.
REM
REM The convert tool recognises each annotation file from its extension and
REM content, and uses any imagery found next to it (images in the same folder,
REM or a video) for the frame names, frame count and timing of the output. Run
REM "viame convert --list-formats" for the readers and writers available.

SET VIAME_INSTALL=.\..\..

SET INPUT_FOLDER=..\object_detector_training\training_data_mouss
SET OUTPUT_FOLDER=example_output
SET OUTPUT_FORMAT=coco
SET DEFAULT_FRAME_RATE=5

CALL "%VIAME_INSTALL%\setup_viame.bat"

viame.exe convert "%INPUT_FOLDER%" "%OUTPUT_FOLDER%" ^
  -o %OUTPUT_FORMAT% --frame-rate %DEFAULT_FRAME_RATE%

PAUSE
