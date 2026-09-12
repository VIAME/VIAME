# algorithm_framework tests

project( test_viame_algorithm_framework )

include( kwiver-test-setup )

###
# The out-of-tree plugin hook
#
# A MODULE rather than a library, because that is what an out-of-tree plugin
# is: something loaded by name at run time, linked against VIAME's headers
# and nothing of VIAME's build.
##
add_library( viame_test_external_plugin MODULE
  external_plugin.cxx
  )

target_link_libraries( viame_test_external_plugin
  PRIVATE    kwiver::vital_vpm
             kwiver::vital_config
  )

set_target_properties( viame_test_external_plugin PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )

kwiver_discover_gtests( viame external_plugins
  LIBRARIES kwiver::vital
            kwiver::vital_vpm
            kwiver::vital_config
            viame_registry
            GTest::gtest )

add_dependencies( test-viame-external_plugins viame_test_external_plugin )

target_compile_definitions( test-viame-external_plugins PRIVATE
  "VIAME_TEST_EXTERNAL_PLUGIN=\"$<TARGET_FILE:viame_test_external_plugin>\""
  # A real shared library that is not a plugin: VIAME's own configuration
  # library, which exports no `viame_register_plugin`.
  "VIAME_TEST_NOT_A_PLUGIN=\"$<TARGET_FILE:vital_config>\"" )
