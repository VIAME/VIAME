#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Run pipeline (requires Ollama to be running, and: ollama pull qwen3-vl:8b)

viame ${VIAME_INSTALL}/configs/pipelines/utility_text_query_ollama_vlm_tracking.pipe \
      -s input:video_filename=input_list.txt \
      -s detector:detector:ollama_vlm:text_query="fish"
