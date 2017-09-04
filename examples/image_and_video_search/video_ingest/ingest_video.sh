#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

source ~/Dev/viame/build/install/setup_viame.sh

# Run pipeline

pipeline_runner -p configs/ingest_video.pipe
