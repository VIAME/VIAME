#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Run HuggingFace zero-shot detector
#
# This detector uses a pretrained Grounding DINO model to detect objects
# from text descriptions without any task-specific training data.
# Useful for quick exploration or bootstrapping annotations.
#
# The text query can be changed by overriding the detector parameter:
#   -s detector:detector:hf_zeroshot:text_queries=fish.seal.whale

viame ${VIAME_INSTALL}/configs/pipelines/detector_huggingface_zeroshot.pipe \
      -s input:video_filename=input_image_list_small_set.txt
