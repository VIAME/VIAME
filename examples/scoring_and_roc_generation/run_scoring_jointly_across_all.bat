@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run score tracks on data for singular metrics

score_tracks.exe                          ^
  --hadwav                                ^
  --computed-tracks detections.kw18       ^
  --truth-tracks groundtruth.kw18 --fn2ts

REM Generate ROC

score_events.exe                          ^
 --computed-tracks detections.kw18        ^
 --truth-tracks groundtruth.kw18          ^
 --fn2ts --kw19-hack --gt-prefiltered     ^
 --ct-prefiltered                         ^
 --roc-dump roc.plot

REM Plot ROC

python.exe plotroc.py roc.plot

pause
