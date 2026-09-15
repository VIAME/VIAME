# algorithm_framework tests

project( test_viame_algorithm_framework )

include( viame-test-setup )

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

# Through libviame in a folded build: a module carrying its own copy of the
# registry would register into a registry the test never looks at.
viame_target_link_libraries( viame_test_external_plugin
  PRIVATE    kwiver::vital_vpm
             kwiver::vital_config
  )

set_target_properties( viame_test_external_plugin PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )

viame_discover_gtests( viame external_plugins
  LIBRARIES kwiver::vital
            kwiver::vital_vpm
            kwiver::vital_config
            viame_registry
            GTest::gtest )

add_dependencies( test-viame-external_plugins viame_test_external_plugin )

target_compile_definitions( test-viame-external_plugins PRIVATE
  "VIAME_TEST_EXTERNAL_PLUGIN=\"$<TARGET_FILE:viame_test_external_plugin>\""
  # A real shared library that is not a plugin: libviame itself, which
  # exports no `viame_register_plugin`. It was `vital_config` until that was
  # folded into it.
  "VIAME_TEST_NOT_A_PLUGIN=\"$<TARGET_FILE:viame_library>\"" )

###
# The logger
#
# Recorded before P8-T04 replaces it. The line the logger writes is what
# anything outside VIAME sees of a run, so the tests are held to the format
# rather than to an opinion about it.
##
viame_discover_gtests( viame logger
  LIBRARIES kwiver::vital_logger
            GTest::gtest )

###
# The file sink
#
# `VIAME_LOG_FILE` is read once, when the first line is written, so a test
# that sets it part way through a process would be testing nothing. This runs
# the same binary again with it already set; every other test in it skips.
##
add_test(
  NAME viame:logger:file_sink
  COMMAND "$<TARGET_FILE:test-viame-logger>"
          "--gtest_filter=logger.the_file_sink_gets_what_stderr_gets"
  )

set_tests_properties( viame:logger:file_sink PROPERTIES
  LABELS "UNIT"
  ENVIRONMENT "VIAME_LOG_FILE=${CMAKE_CURRENT_BINARY_DIR}/file_sink.log" )

###
# The file system helpers
#
# Recorded before P8-T05 puts `std::filesystem` underneath them. The
# expectations were taken by running kwiversys, not read out of its
# documentation; four of them are places the standard library answers
# differently.
#
# It writes directories, so it runs somewhere it is allowed to.
##
viame_discover_gtests( viame file_system
  LIBRARIES kwiver::vital_util
            GTest::gtest )

# It makes and removes directories relative to the working directory, so it
# is given one of its own. The test does the `chdir` itself rather than ctest
# doing it, because a `WORKING_DIRECTORY` set from `TEST_INCLUDE_FILES` would
# be evaluated at test time, when there is no `CMAKE_CURRENT_BINARY_DIR`.
target_compile_definitions( test-viame-file_system PRIVATE
  "VIAME_FS_TEST_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/file_system_scratch\"" )

###
# The regular expression the parsers use
#
# Recorded before P8-T05 changed the engine from kwiversys's POSIX one to
# `std::regex`. Three of the four patterns read configuration a user wrote.
##
viame_discover_gtests( viame regex
  LIBRARIES kwiver::vital_util
            GTest::gtest )

###
# `config_block`
#
# Recorded before P8-T09, whose instruction is "keep config_block semantics"
# while removing what nothing uses -- which needs the semantics written down
# first. Every pipeline, algorithm and applet in VIAME reads its parameters
# through this class and nothing in the tree said what it does.
##
viame_discover_gtests( viame config_block
  LIBRARIES kwiver::vital_config
            kwiver::vital_exceptions
            kwiver::vital_util
            GTest::gtest )

###
# The python library finder
#
# What the python module loader preloads libpython with when `PYTHON_LIBRARY`
# is not set, which since P10-T01 is always. It returned nothing on Debian
# and Ubuntu pythons; the test fakes their layout and a CPython built into
# the install, and loads the module from the source tree.
##
if( VIAME_ENABLE_PYTHON )
  viame_add_test(
    NAME    "unit:algorithm_framework:find_python_library"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_find_python_library.py"
    LABELS  UNIT
    TIMEOUT 120
  )
endif()
