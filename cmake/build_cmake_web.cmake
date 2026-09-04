# VIAME CMake Web Configuration
# Additional settings for VIAME-Web builds (use with viame_cmake_docker.cmake)
#
# Usage: cmake -C viame_cmake_base.cmake -C viame_cmake_docker.cmake -C viame_cmake_web.cmake ...

# Web-specific settings
set(VIAME_ENABLE_WEB_EXCLUDES ON CACHE BOOL "Exclude desktop-only components")

# Model downloads - minimal for web, save the two packs the CRITICAL tests
# exercise. Five of the seven drive detector_generic_proposals,
# detector_default_fish_no_motion, tracker_generic_proposals,
# tracker_default_fish_fusion and measurement_default_fish_fully_auto, so
# without these packs those pipelines are absent and the suite cannot gate
# anything. Turn them back off only if the suite is scoped down to match.
set(VIAME_DOWNLOAD_MODELS ON CACHE BOOL "Download models")
set(VIAME_DOWNLOAD_MODELS-DEFAULT-FISH ON CACHE BOOL "Download fish models")
set(VIAME_DOWNLOAD_MODELS-GENERIC ON CACHE BOOL "Download generic models")
set(VIAME_DOWNLOAD_MODELS-PYSOT OFF CACHE BOOL "Download PySoT models")
set(VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL OFF CACHE BOOL "Download Arctic seal models")
set(VIAME_DOWNLOAD_MODELS-HABCAM OFF CACHE BOOL "Download HabCam models")
set(VIAME_DOWNLOAD_MODELS-MOUSS OFF CACHE BOOL "Download MOUSS models")
