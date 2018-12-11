
#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL=./../..

source ${VIAME_INSTALL}/setup_viame.sh

# Run score tracks on data for singular metrics

score_tracks --hadwav \
  --computed-tracks detections.csv --computed-format noaa-csv \
  --truth-tracks groundtruth.csv --truth-format noaa-csv \
  --fn2ts > score_tracks_output.txt

# Generate ROC

score_events \
  --computed-tracks detections.csv --computed-format noaa-csv \
  --truth-tracks groundtruth.csv --truth-format noaa-csv \
  --fn2ts --kw19-hack --gt-prefiltered --ct-prefiltered \
  --roc-dump roc.plot

# Plot ROC

python plotroc.py roc.plot
