###
# This file contains an assortment of VIAME standalone tools
##

###
# Python standalone tools
##
set( PYTHON_SCRIPTS
  add_ons.py
  segment.py
  convert.py
  extract.py
  inspect_file.py
  index.py
  view.py
  search.py
  pipeline.py
  plot.py
  run.py
  metadata.py
  ensemble.py
  monitor.py
)

if( VIAME_ENABLE_OPENCV )
  set( PYTHON_SCRIPTS
    ${PYTHON_SCRIPTS}
    calibrate.py
    depth.py
    disparity.py
    mosaic.py
    register.py
    3d.py
    rectify.py
  )
endif()

if( VIAME_ENABLE_PYTORCH )
  set( PYTHON_SCRIPTS
    ${PYTHON_SCRIPTS}
    gpu.py
  )
endif()

###
# C/C++ compiled standalone tools
##

# VIAME tool runner - main entry point similar to kwiver
kwiver_add_executable( viame
    viame.cxx
  )

target_link_libraries( viame
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_config
               kwiver::vital_exceptions
               kwiver::vital_logger
               kwiver::vital_util
               kwiver::kwiversys
               kwiver::vital_applets
  )

# Only the applet attribute names are needed from core; linking it would load
# its every dependency on each viame invocation
target_include_directories( viame
  PRIVATE      ${VIAME_SOURCE_DIR}/plugins/core
  )

###
# The kwiver tool runner
#
# Imported from kwiver's `tools/` in P5-T05, unchanged. `kwiver runner` is
# what the golden pipeline runner and the annotator launcher invoke, so the
# executable and its name have to survive the submodule.
##
kwiver_add_executable( kwiver
    kwiver_tool_runner.cxx
  )

target_link_libraries( kwiver
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_logger
               kwiver::kwiversys
               kwiver::vital_applets
  )

###
# VIAME tool applets plugin - train and other applets
##
set( applet_sources
  register_applets.cxx
  csv.cxx
  configs.cxx
  convert.cxx
  json.cxx
  pipe_check.cxx
  registry_dump.cxx
  resample.cxx
  run.cxx
  score.cxx
  train.cxx
)

set( applet_headers
  csv.h
  configs.h
  convert.h
  json.h
  pipe_check.h
  registry_dump.h
  resample.h
  run.h
  score.h
  train.h
)

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin( viame_tools_applets
  SUBDIR       viame/applets
  SOURCES      ${applet_sources} ${applet_headers}
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_config
               kwiver::vital_exceptions
               kwiver::vital_logger
               kwiver::vital_algo
               kwiver::vital_util
               kwiver::kwiversys
               kwiver::vital_applets
               kwiver::sprokit_pipeline
               kwiver::sprokit_pipeline_util
               kwiver::kwiver_adapter
               viame_core
               viame_claude
)

# The json applet parses with the rapidjson vendored inside KWIVER's cereal,
# whose headers include each other relative to this directory
if( NOT KWIVER_SOURCE_DIR )
  set( KWIVER_SOURCE_DIR "${VIAME_PACKAGES_DIR}/kwiver" )
endif()
target_include_directories( viame_tools_applets
  PRIVATE      ${VIAME_SOURCE_DIR}/third_party/cereal
               ${VIAME_SOURCE_DIR}/third_party/cereal/cereal/external
  )

# pipe_bakery.h includes its neighbours by bare name and only some of them
# are installed, so the sprokit sources have to be reachable that way too
target_include_directories( viame_tools_applets
  PRIVATE      ${KWIVER_SOURCE_DIR}/sprokit/src
               ${KWIVER_SOURCE_DIR}/sprokit/src/sprokit/pipeline_util
  )

if( VIAME_ENABLE_PYTHON )
  target_compile_definitions( viame_tools_applets
    PRIVATE VIAME_TOOLS_ENABLE_PYTHON )
endif()

# The applets this gates are python scripts that import cv2, a wheel; nothing
# on the C++ side of the tools links OpenCV any more, since `viame score`
# handed its plot rendering to matplotlib in P7-T07
if( VIAME_ENABLE_OPENCV )
  target_compile_definitions( viame_tools_applets
    PRIVATE VIAME_TOOLS_HAVE_OPENCV )
endif()

if( VIAME_ENABLE_PYTORCH )
  target_compile_definitions( viame_tools_applets
    PRIVATE VIAME_TOOLS_HAVE_PYTORCH )
endif()

if( VIAME_ENABLE_PYTHON )
  install( FILES       ${PYTHON_SCRIPTS}
           DESTINATION configs )
endif()

