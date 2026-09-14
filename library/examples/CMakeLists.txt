# Examples and templates
#
# The sprokit process template, imported from kwiver's
# `sprokit/processes/examples/process_template` in P5-T04. It registers a
# `template` process, which is part of the registry contract, and it is what
# someone copies to start a new process, so the `//++` commentary is the
# point of the files and is left alone.
#
# `algo_wrapper_process.{h,cxx}` did not come across: kwiver's own CMakeLists
# never named them, so they had not compiled in years.

# Sprokit's own worked examples: twenty-one processes and two clusters, all
# of them in the registry baseline. Imported from `sprokit/src/processes` in
# P5-T05.
add_subdirectory( processes )

set( process_headers
  template_algo_wrapper.h
  template_process.h
  template_type_traits.h
  )

set( process_sources
  register_processes.cxx
  template_algo_wrapper.cxx
  template_process.cxx
  )

viame_add_plugin( viame_processes_examples
  KIND        PROCESSES
  SOURCES          ${process_sources}
                   ${process_headers}
  PRIVATE          kwiver::sprokit_pipeline
                   kwiver::vital
                   kwiver::vital_algo
                   kwiver::vital_vpm
                   kwiver::vital_logger
                   kwiver::vital_config
                   kwiver::vital_util
                   kwiver::vital_exceptions
                   viame_image_ops
  )

target_include_directories( viame_processes_examples PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

install( FILES       README.rst
                     ${process_headers}
                     ${process_sources}
                     CMakeLists.txt
         DESTINATION examples/process_template )

###
# Algorithm examples
#
# `hello_world_detector` and `hello_world_filter`, from the `examples` plugin in
# P2-T07: the smallest complete C++ algorithms, each one `PLUGGABLE_IMPL`
# and a method, and the python processes of the same names. `templates/`
# holds the detector templates someone copies to start a new algorithm; they
# carry `@template@` placeholders and are not built.
##

set( plugin_headers
  hello_world_detector.h
  hello_world_filter.h
  )

set( plugin_sources
  hello_world_detector.cxx
  hello_world_filter.cxx
  )

viame_install_headers(
  SUBDIR     viame/examples
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_examples_export.h
  NOPATH   SUBDIR     viame/examples
  )

viame_add_library( viame_examples
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_examples PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_examples
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
  )

set_target_properties( viame_examples PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_examples
  register_algorithms.cxx
  )

if( VIAME_ENABLE_PYTHON )
  viame_add_python_package( examples EXCLUDE "^(processes|templates)/" )
endif()
