# VIAME Plugin Tests

add_subdirectory( core )

if( VIAME_ENABLE_OPENCV )
  add_subdirectory( opencv )
endif()

if( VIAME_ENABLE_PYTORCH )
  add_subdirectory( pytorch )
endif()

# Every tracker trainer loads and survives a configuration round trip. Cheap
# enough to be worth running whatever else is enabled; trainers whose
# subsystem is not built skip themselves.
if( VIAME_ENABLE_PYTHON )
  add_test(
    NAME viame_plugins:tracker_trainers_load
    COMMAND bash -c
      "source ${VIAME_BUILD_INSTALL_PREFIX}/setup_viame.sh && \
       python -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_tracker_trainers_load.py \
       -v --tb=short"
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  set_tests_properties( viame_plugins:tracker_trainers_load
    PROPERTIES
      LABELS "UNIT;CORE;PYTORCH"
      TIMEOUT 300
  )
endif()
