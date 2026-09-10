# The `kwiver::` names for kwiver's targets
#
# Kwiver's installed config package exported its libraries under a namespace;
# since P1-T05 kwiver is a subdirectory of this build instead, and a
# subdirectory's targets have no namespace. VIAME's CMakeLists name them the
# way they always did, so each gets an alias.
#
# This is the whole list VIAME names out of what `packages/kwiver` still
# builds. `sprokit_pipeline`, `sprokit_pipeline_util` and `kwiver_adapter`
# are VIAME's own targets since P5-T05 and alias themselves where they are
# defined, in `library/pipeline_framework`. Adding a name here without a
# `kwiver::` use somewhere is how the list stops meaning anything.

foreach( _viame_kwiver_target
         kwiversys
         vital
         vital_algo
         vital_applets
         vital_config
         vital_exceptions
         vital_logger
         vital_util
         vital_vpm )
  if( NOT TARGET "${_viame_kwiver_target}" )
    message( FATAL_ERROR
      "kwiver did not define the target ${_viame_kwiver_target}" )
  endif()

  if( NOT TARGET "kwiver::${_viame_kwiver_target}" )
    add_library( "kwiver::${_viame_kwiver_target}"
                 ALIAS "${_viame_kwiver_target}" )
  endif()
endforeach()
