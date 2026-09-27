# These tests configure and source setup scripts without a VIAME build.
if( UNIX )
  foreach( shell bash zsh )
    find_program( VIAME_${shell}_EXECUTABLE NAMES ${shell} )
    if( VIAME_${shell}_EXECUTABLE )
      add_test( NAME test-setup-${shell}
        COMMAND "${CMAKE_COMMAND}"
          "-DSHELL_EXECUTABLE=${VIAME_${shell}_EXECUTABLE}"
          "-DVIAME_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
          "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/${shell}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/test_setup_script.cmake" )
      set_tests_properties( test-setup-${shell} PROPERTIES LABELS "SETUP" )
    endif()
  endforeach()
endif()
