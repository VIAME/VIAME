#!/bin/sh

# Configurable Input Paths
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/.."

# Other paths generated from root
export VIAME_PIPELINES=${VIAME_INSTALL}/configs/pipelines/

# Remove sea lion pipelines
rm -rf ${VIAME_PIPELINES}/arctic*fusion*
