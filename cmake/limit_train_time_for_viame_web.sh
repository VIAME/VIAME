#!/bin/sh

# Limit training time for VIAME Web deployments
# Sets timeout to 5 days (432000 seconds) in all training configs

# Configurable Input Paths
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/.."

# Other paths generated from root
export VIAME_PIPELINES=${VIAME_INSTALL}/configs/pipelines/

# 5 days in seconds: 5 * 24 * 60 * 60 = 432000
TIMEOUT_SECONDS=432000

echo "Setting training timeout to 5 days (${TIMEOUT_SECONDS} seconds) in all configs..."

# Update RF-DETR configs
for conf in ${VIAME_PIPELINES}/train_detector_rf_detr*.conf; do
  if [ -f "$conf" ]; then
    sed -i "s/rf_detr:timeout = .*/rf_detr:timeout = ${TIMEOUT_SECONDS}/" "$conf"
    echo "  Updated: $conf"
  fi
done

# Update MIT-YOLO configs
for conf in ${VIAME_PIPELINES}/train_detector_mit_yolo*.conf; do
  if [ -f "$conf" ]; then
    sed -i "s/mit_yolo:timeout = .*/mit_yolo:timeout = ${TIMEOUT_SECONDS}/" "$conf"
    echo "  Updated: $conf"
  fi
done

# Update Netharn configs (handles both numeric and 'default' values)
for conf in ${VIAME_PIPELINES}/train_detector_netharn*.conf; do
  if [ -f "$conf" ]; then
    sed -i "s/netharn:timeout = .*/netharn:timeout = ${TIMEOUT_SECONDS}/" "$conf"
    echo "  Updated: $conf"
  fi
done

echo "Done."
