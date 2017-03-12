@echo off

score_events \
--computed-tracks detections.kw18 \
--truth-tracks fish_gt.kw18 --fn2ts --a PersonDigging \
--kw19-hack --act-ext-ct all --gt-prefiltered --roc-dump roc.plot

score_tracks \
  --hadwav \
  --computed-tracks detections.kw18 \
  --truth-tracks groundtruth.kw18 --fn2ts

pause
