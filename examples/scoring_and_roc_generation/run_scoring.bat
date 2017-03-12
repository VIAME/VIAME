@echo off

SET VIAME_INSTALL=%~dp0\..\..\

%VIAME_INSTALL%\bin\score_events.exe      ^
 --computed-tracks detections.kw18        ^
 --truth-tracks fish_gt.kw18 --fn2ts      ^
 --a PersonDigging --kw19-hack            ^
 --act-ext-ct all --gt-prefiltered        ^
 --roc-dump roc.plot

%VIAME_INSTALL%\bin\score_tracks.exe      ^
  --hadwav                                ^
  --computed-tracks detections.kw18       ^
  --truth-tracks groundtruth.kw18 --fn2ts

pause
