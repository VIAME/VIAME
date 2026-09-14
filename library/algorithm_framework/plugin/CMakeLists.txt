###
# The plugin registry
#
# `registry` is the store -- factories by interface name -- and
# `plugin_manager` the singleton around it. Neither loads anything since
# P8-T03: the registration functions are linked in and called by
# `library/algorithm_framework/registry`.

###
# vital plugin manager
set(vital_vpm_headers_public
  pluggable.h
  plugin_factory.h
  registry.h
  plugin_manager.h
  plugin_registrar.h
  register_algorithm.h
#  plugin_info.h
  )

set(vital_vpm_headers_private
    #  plugin_manager_internal.h
    ${CMAKE_CURRENT_BINARY_DIR}/vital_vpm_export.h
    )

set(vital_vpm_sources
  plugin_factory.cxx
  registry.cxx
  plugin_manager.cxx
  )

# This has to be built shared due to the use of the singleton pattern holding
# the central manager instance.

viame_add_library(vital_vpm SHARED
  ${vital_vpm_headers_public}
  ${vital_vpm_headers_private}
  ${vital_vpm_sources}
  )


viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_vpm_export.h
  SUBDIR      viame/algorithm_framework/plugin
  NOPATH
  )

target_link_libraries(vital_vpm
  PUBLIC              vital_util
  PRIVATE             vital_logger
                      vital_exceptions
                      ${VITAL_BOOST_REGEX}
  )

###
# Add tests if enbled

# Republished under the prefix the imported sources use.
viame_lite_generated( vital_vpm_export.h algorithm_framework/plugin )
