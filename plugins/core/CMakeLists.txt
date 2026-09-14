# Build / Install Plugin containing core algorithm implementations

set( plugin_headers
  windowed_trainer.h
)

set( plugin_sources
  windowed_trainer.cxx
)

viame_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_core_export.h
  NOPATH   SUBDIR     viame
  )

viame_add_library( viame_core
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

set( CORE_LINK_LIBRARIES
  viame_utilities
  kwiver::vital
  kwiver::vital_algo
  kwiver::vital_applets
  kwiver::vital_config
  kwiver::vital_exceptions
  kwiver::vital_logger
  kwiver::vital_util
  viame_measurement
  # `windowed_utils` moved to `library/object_detectors` when darknet did:
  # `library/` cannot reach up into `plugins/`, and darknet's detector needs
  # the chipping
  viame_object_detectors
  kwiver::sprokit_pipeline
  kwiver::kwiver_adapter
  )

# The polygon rasterisation the CSV parsers use is in-house now, and so is
# the OpenCV YAML `camera_rig_io` reads calibrations with (P7-T05)
# ... and the image writer the epipolar debug overlay uses (P7-T06)
set( AUX_LINK_LIBRARIES
     viame_image_ops viame_file_io viame_video_io ${AUX_LINK_LIBRARIES} )

# `plugins/core` has no OpenCV left as of P7-T07: the CSV writers' polygon
# conversion was the last of it, and `image_ops::find_borders` is what it is
# now. The `find_package` and the `VIAME_ENABLE_OPENCV` definition that stood
# here are gone with it.

# `viame_core` linked libpython for the DINO feature matching in the stereo
# pairing, through the Python C API. That code is `library/measurement` since
# P2-T06 and carries the link itself; nothing left here, or in anything that
# links this, includes `Python.h` or tests `VIAME_ENABLE_PYTHON` -- the IQR
# header only names it in a comment -- so the library everything links is
# free of libpython again.

target_link_libraries( viame_core
  PUBLIC               ${CORE_LINK_LIBRARIES}
                       ${AUX_LINK_LIBRARIES}
  )

# Propagate compile definitions to dependent targets (important for ABI compatibility)
if( AUX_COMPILE_DEFINITIONS )
  target_compile_definitions( viame_core PUBLIC ${AUX_COMPILE_DEFINITIONS} )
endif()

set_target_properties( viame_core PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_core
  register_algorithms.cxx
  )

if( VIAME_ENABLE_PYTHON )

  viame_create_python_init( core )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    core
    __init__ )

endif()

