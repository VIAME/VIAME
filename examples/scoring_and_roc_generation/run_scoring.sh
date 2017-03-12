

/home/matt/Dev/vidtk/build/bin/score_events \
--computed-tracks yolo_v2_detections.kw18 \
--truth-tracks fish_gt.kw18 --fn2ts --a PersonDigging \
--kw19-hack --act-ext-ct all --gt-prefiltered --roc-dump roc.plot

score_tracks \
  --hadwav \
  --computed-tracks yolo_v2_detections.kw18 \
  --truth-tracks fish_gt.kw18 --fn2ts
