#!/bin/bash

# Build a VIAME Vertex AI container with a custom add-on and default pipeline.
#
# Usage:
#   ./vertex_ai_docker_with_add_on.sh <addon.zip> <pipeline> <image-name> [output-type]
#
# Example:
#   ./vertex_ai_docker_with_add_on.sh VIAME-DEFAULT-FISH-Models.zip \
#     pipelines/detector_default_fish.pipe \
#     viame-vertex-fish:latest \
#     coco

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIAME_SRC="$(dirname "$SCRIPT_DIR")"

usage() {
  echo "Usage: $0 <addon-zip> <pipeline> <image-name> [output-type]"
  echo ""
  echo "Arguments:"
  echo "  addon-zip     Path to a VIAME add-on .zip file"
  echo "  pipeline      Default pipeline to run (e.g. pipelines/detector_default_fish.pipe)"
  echo "  image-name    Name (and optional tag) for the output Docker image"
  echo "  output-type   Output format: coco, viame_csv, kw18 (default: coco)"
  exit 1
}

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  usage
fi

ADDON_ZIP="$(realpath "$1")"
PIPELINE="$2"
IMAGE_NAME="$3"
OUTPUT_TYPE="${4:-coco}"

if [ ! -f "$ADDON_ZIP" ]; then
  echo "Error: Add-on zip not found: $ADDON_ZIP"
  exit 1
fi

# Create a temporary build context
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

# Copy the vertex-ai plugin sources and the add-on zip into the build context
mkdir -p "$BUILD_DIR/plugins/vertex-ai"
cp "$VIAME_SRC/plugins/vertex-ai/"*.py "$BUILD_DIR/plugins/vertex-ai/"
cp "$ADDON_ZIP" "$BUILD_DIR/addon.zip"

# Generate a Dockerfile that layers the add-on on top of the base vertex image
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

# Install the add-on (zip extracts configs/pipelines/* into the VIAME install)
COPY addon.zip /tmp/addon.zip
RUN python -c "import zipfile; zipfile.ZipFile('/tmp/addon.zip').extractall('/opt/noaa/viame')" \
    && rm /tmp/addon.zip

WORKDIR /workspace

ENV VIAME_INSTALL_DIR=/opt/noaa/viame
ENV VIAME_WORK_DIR=/workspace
ENV AIP_HTTP_PORT=8080
DOCKERFILE

# Append the pipeline and output type ENVs using the caller's values
echo "ENV VIAME_PIPELINE=$PIPELINE" >> "$BUILD_DIR/Dockerfile"
echo "ENV VIAME_OUTPUT_TYPE=$OUTPUT_TYPE" >> "$BUILD_DIR/Dockerfile"

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
ls -lh "$BUILD_DIR"
echo ""
echo "=== Dockerfile ==="
cat "$BUILD_DIR/Dockerfile"
echo ""
echo "=== Building $IMAGE_NAME ==="

docker build -t "$IMAGE_NAME" "$BUILD_DIR"

echo ""
echo "Done. Run locally with:"
echo "  docker run --gpus all -p 8080:8080 -v /path/to/videos:/workspace/input $IMAGE_NAME"
