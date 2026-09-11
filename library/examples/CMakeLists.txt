# Plugin templates
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

kwiver_add_plugin( viame_processes_examples
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
                   ${process_headers}
  PRIVATE          kwiver::sprokit_pipeline
                   kwiver::kwiversys
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
