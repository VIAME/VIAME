@echo off

SET VIAME_INSTALL=%~dp0\..\..\

REM Run score tracks on data for singular metrics

%VIAME_INSTALL%\bin\score_tracks.exe      ^
  --hadwav                                ^
  --computed-tracks detections.kw18       ^
  --truth-tracks groundtruth.kw18 --fn2ts

REM Generate ROC

%VIAME_INSTALL%\bin\score_events.exe      ^
 --computed-tracks detections.kw18        ^
 --truth-tracks groundtruth.kw18          ^
 --fn2ts --kw19-hack --gt-prefiltered     ^
 --ct-prefiltered                         ^
 --roc-dump roc.plot

REM Plot ROC

python.exe plotroc.py roc.plot

pause
