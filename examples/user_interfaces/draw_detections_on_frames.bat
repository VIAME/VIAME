@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)
SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline
kwiver.exe runner "%VIAME_INSTALL%\configs\pipelines\filter_draw_dets.pipe" ^
  -s input:video_filename=example_image_list.txt ^
  -s detection_reader:file_name=example_detections.csv
