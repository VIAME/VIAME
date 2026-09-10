
set(__testable_modnames
  test_serialize_json
  )

# All python filenames_we in the vital.tests module
set(__all_modnames
  __init__
  serialize_json_utils
  ${__testable_modnames}
  )

foreach (modname IN LISTS __all_modnames)
  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/${modname}.py
    arrows/tests
    ${modname})
endforeach()
