#!/bin/bash

# Build a VIAME Vertex AI container with one or more custom add-ons.
#
# Usage:
#   ./viame_gpu_vertex_ai_with_addon.sh <addon-zips> <pipeline> <image-name> [output-type]
#
# Examples:
#   # Single add-on:
#   ./viame_gpu_vertex_ai_with_addon.sh VIAME-DEFAULT-FISH-Models.zip \
#     pipelines/detector_default_fish.pipe \
#     viame-vertex-fish:latest
#
#   # Multiple add-ons (comma-separated):
#   ./viame_gpu_vertex_ai_with_addon.sh VIAME-DEFAULT-FISH-Models.zip,VIAME-GENERIC-Models.zip \
#     pipelines/tracker_default_fish.pipe \
#     viame-vertex-fish:latest \
#     coco

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIAME_SRC="$(dirname "$SCRIPT_DIR")"

usage() {
  echo "Usage: $0 <addon-zips> <pipeline> <image-name> [output-type]"
  echo ""
  echo "Arguments:"
  echo "  addon-zips    Comma-separated paths to VIAME add-on .zip files"
  echo "  pipeline      Default pipeline to run (e.g. pipelines/detector_default_fish.pipe)"
  echo "  image-name    Name (and optional tag) for the output Docker image"
  echo "  output-type   Output format: coco, viame_csv, kw18 (default: coco)"
  exit 1
}

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  usage
fi

ADDON_ZIPS="$1"
PIPELINE="$2"
IMAGE_NAME="$3"
OUTPUT_TYPE="${4:-coco}"

# Create a temporary build context
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

# Copy the vertex-ai plugin sources into the build context
mkdir -p "$BUILD_DIR/plugins/vertex-ai" "$BUILD_DIR/addons"
cp "$VIAME_SRC/plugins/vertex-ai/"*.py "$BUILD_DIR/plugins/vertex-ai/"

# Copy and validate each add-on zip
IFS=',' read -ra ZIP_LIST <<< "$ADDON_ZIPS"
ADDON_IDX=0
for zip_path in "${ZIP_LIST[@]}"; do
  zip_path="$(echo "$zip_path" | xargs)"  # trim whitespace
  zip_path="$(realpath "$zip_path")"
  if [ ! -f "$zip_path" ]; then
    echo "Error: Add-on zip not found: $zip_path"
    exit 1
  fi
  cp "$zip_path" "$BUILD_DIR/addons/addon_${ADDON_IDX}.zip"
  ADDON_IDX=$((ADDON_IDX + 1))
done

echo "Installing ${ADDON_IDX} add-on(s)"

# Generate a Dockerfile that layers the add-ons on top of the base vertex image
cat > "$BUILD_DIR/Dockerfile" <<'DOCKERFILE'
FROM kitware/viame:gpu-algorithms-web

# Vertex AI Flask and GCP dependencies
RUN pip install --no-cache-dir \
  "flask>=2.3" \
  "gunicorn>=21.2" \
  "google-cloud-storage>=2.10" \
  "google-cloud-aiplatform>=1.38"

# Copy vertex-ai app into the VIAME configs directory
COPY plugins/vertex-ai/*.py /opt/noaa/viame/configs/

# Install add-on(s) (each zip extracts configs/pipelines/* into the VIAME install)
COPY addons/ /tmp/addons/
RUN for z in /tmp/addons/*.zip; do \
      python -c "import zipfile; zipfile.ZipFile('${z}').extractall('/opt/noaa/viame')"; \
    done && rm -rf /tmp/addons

WORKDIR /workspace

ENV VIAME_INSTALL_DIR=/opt/noaa/viame
ENV VIAME_WORK_DIR=/workspace
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH}
ENV AIP_HTTP_PORT=8080
DOCKERFILE

# Append the pipeline and output type ENVs using the caller's values
echo "ENV VIAME_PIPELINE=$PIPELINE" >> "$BUILD_DIR/Dockerfile"
echo "ENV VIAME_OUTPUT_TYPE=$OUTPUT_TYPE" >> "$BUILD_DIR/Dockerfile"
echo "ENV VIAME_FRAME_RATE=10" >> "$BUILD_DIR/Dockerfile"

cat >> "$BUILD_DIR/Dockerfile" <<'DOCKERFILE'

EXPOSE 8080

ENTRYPOINT ["bash", "-c", \
  "source /opt/noaa/viame/setup_viame.sh && \
   cd /opt/noaa/viame/configs && \
   gunicorn --bind 0.0.0.0:${AIP_HTTP_PORT} \
            --timeout 600 \
            --workers 1 \
            --threads 4 \
            app:app"]
DOCKERFILE

echo "=== Build context ==="
ls -lh "$BUILD_DIR" "$BUILD_DIR/addons/"
echo ""
echo "=== Dockerfile ==="
cat "$BUILD_DIR/Dockerfile"
echo ""
echo "=== Building $IMAGE_NAME ==="

docker build -t "$IMAGE_NAME" "$BUILD_DIR"

echo ""
echo "Done. Run locally with:"
echo "  docker run --gpus all -p 8080:8080 -v /path/to/videos:/workspace/input $IMAGE_NAME"
