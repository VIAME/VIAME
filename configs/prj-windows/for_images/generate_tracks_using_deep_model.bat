@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=C:\Program Files\VIAME

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

pipeline_runner.exe -p "%VIAME_INSTALL%\configs\pipelines\tracker_default.tut.pipe" ^
                    -s input:video_filename=input_list.txt ^
                    -s detector:detector:darknet:net_config=deep_training\yolo_v2.cfg ^
                    -s detector:detector:darknet:weight_file=deep_training\models\yolo_v2.backup ^
                    -s detector:detector:darknet:class_names=deep_training\yolo_v2.lbl ^
                    -s detector:detector:darknet:scale=1.4 ^
                    -s detector_writer:file_name=deep_detections.csv ^
                    -s track_writer:file_name=deep_tracks.csv

pause
