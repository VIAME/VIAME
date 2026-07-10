@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run stabilized IOU tracker for moving camera scenarios
REM Note: Requires a domain-specific add-on (e.g. default-fish) that includes
REM the stabilized IOU tracker pipeline with a detector.

viame "%VIAME_INSTALL%\configs\pipelines\tracker_stabilized_iou.pipe" ^
      -s input:video_filename=input_list.txt

PAUSE
