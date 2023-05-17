cmake_minimum_required(VERSION 3.11)
project(ifremer_tk
  VERSION 0.1
  LANGUAGES CXX
  )

file(GLOB PIPELINE_FILES "${PROJECT_SOURCE_DIR}/pipelines/*.*")
file(GLOB MODEL_FILES "${PROJECT_SOURCE_DIR}/pipelines/models/*.zip")

install(FILES
  ${PIPELINE_FILES}
  DESTINATION configs/pipelines
  )

install(FILES
  ${MODEL_FILES}
  DESTINATION configs/pipelines/models
  )

install(DIRECTORY
  "${PROJECT_SOURCE_DIR}/camera_calibration"
  DESTINATION configs
  )