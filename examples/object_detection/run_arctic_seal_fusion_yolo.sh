 #!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL=./../..

source ${VIAME_INSTALL}/setup_viame.sh 

# Run pipeline

pipeline_runner -p ${VIAME_INSTALL}/configs/pipelines/detector_arctic_seal_fusion_yolo.pipe \
                -s optical_input:video_filename=input_image_list_seal_eo.txt \
                -s thermal_input:video_filename=input_image_list_seal_ir.txt

