# VIAME Tools Tests

if( VIAME_ENABLE_PYTHON )
  viame_add_pytest_test(
    NAME    "tools:viame_applets"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_viame_applets.py"
    LABELS  tools
    TIMEOUT 900
    SOURCE_SETUP
  )
  viame_add_pytest_test(
    NAME    "tools:convert"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_convert.py"
    LABELS  tools
    TIMEOUT 600
    SOURCE_SETUP
  )

  foreach( tool IN ITEMS
    3d
    add_ons
    configs
    csv
    database
    disparity
    ensemble
    extract
    index
    inspect_file
    json
    metadata
    monitor
    mosaic
    pipeline
    plot
    rectify
    resample
    run
    run_bulk
    segment
    viame
  )
    viame_add_pytest_test(
      NAME    "tools:${tool}"
      TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_${tool}.py"
      LABELS  tools
      TIMEOUT 600
      SOURCE_SETUP
    )
  endforeach()
endif()
