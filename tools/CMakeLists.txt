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
viame_add_executable( viame
    viame.cxx
  )

viame_target_link_libraries( viame
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_config
               kwiver::vital_exceptions
               kwiver::vital_logger
               kwiver::vital_util
               kwiver::vital_applets
               viame::registry
  )

###
# The kwiver tool runner
#
# Imported from kwiver's `tools/` in P5-T05, unchanged. `kwiver runner` is
# what the golden pipeline runner and the annotator launcher invoke, so the
# executable and its name have to survive the submodule.
##
viame_add_executable( kwiver
    kwiver_tool_runner.cxx
  )

viame_target_link_libraries( kwiver
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_logger
               kwiver::vital_applets
               viame::registry
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

viame_add_plugin( viame_tools_applets
  KIND        APPLETS
  SOURCES      ${applet_sources} ${applet_headers}
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_config
               kwiver::vital_exceptions
               kwiver::vital_logger
               kwiver::vital_algo
               kwiver::vital_util
               kwiver::vital_applets
               kwiver::sprokit_pipeline
               kwiver::sprokit_pipeline_util
               kwiver::kwiver_adapter
               viame_file_io
               viame_utilities
               viame_training
               viame_evaluation
)

# The json applet parses with the vendored rapidjson, reached through
# `library/file_io/json.h`
if( NOT KWIVER_SOURCE_DIR )
  set( KWIVER_SOURCE_DIR "${VIAME_PACKAGES_DIR}/kwiver" )
endif()
target_include_directories( viame_tools_applets
  PRIVATE      ${VIAME_SOURCE_DIR}/library/tpl/rapidjson
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

  # The Vertex AI custom-container application: a Flask server and its two
  # handlers, which import each other by bare name and run from one
  # directory. From `vertex-ai` in P2-T08, where they installed as a
  # `viame/vertex-ai` package that no import could name.
  if( VIAME_ENABLE_VERTEX_AI )
    install( FILES       vertex_ai/app.py
                         vertex_ai/process_handler.py
                         vertex_ai/train_handler.py
             DESTINATION configs/vertex_ai )
  endif()
endif()


###
# The `viame.tools` python package
#
# `dump_klv` was `kwiver.tools.dump_klv` and lived in `python/kwiver/tools`
# until P11-T02b brought it beside the C++ tools it belongs with.
##
if( VIAME_ENABLE_PYTHON )
  viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/tools_init.py
    tools
    __init__ )
  viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/dump_klv.py
    tools
    dump_klv )

  # `dump_klv` is also a script, and the tests below run it as one.
  configure_file(
    dump_klv.py
    ${KWIVER_BINARY_DIR}/bin/dump_klv.py
    COPYONLY
    )

  if( KWIVER_ENABLE_PYTHON_TESTS )
    add_subdirectory( tests )
  endif()
endif()
