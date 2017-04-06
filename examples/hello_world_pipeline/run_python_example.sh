#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

source ../../setup_viame.sh 

# Run pipeline

pipeline_runner -p hello_world_python.pipe -S pythread_per_process 
