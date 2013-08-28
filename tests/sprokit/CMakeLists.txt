include_directories("${sprokit_source_dir}/src")
include_directories("${sprokit_binary_dir}/src")

function (sprokit_discover_tests group libraries file)
  file(STRINGS "${file}" test_lines)
  set(properties)

  sprokit_build_tooled_test("${group}" "${libraries}" "${file}")

  foreach (test_line IN LISTS test_lines)
    set(test_name)
    set(property)

    string(REGEX MATCH "^IMPLEMENT_TEST\\(([A-Za-z_]+)\\)$"
      match "${test_line}")
    if (match)
      set(test_name "${CMAKE_MATCH_1}")
      sprokit_add_tooled_test("${group}" "${test_name}"
        ${ARGN})
      if (properties)
        set_tests_properties("test-${group}-${test_name}"
          PROPERTIES
            ${properties})
      endif ()
      set(properties)
      set(sprokit_test_environment)
    endif ()
    string(REGEX MATCHALL "^TEST_PROPERTY\\(([A-Za-z_]+), (.*)\\)$"
      match "${test_line}")
    if (match)
      set(prop "${CMAKE_MATCH_1}")
      string(CONFIGURE "${CMAKE_MATCH_2}" prop_value
        @ONLY)
      if (prop STREQUAL "ENVIRONMENT")
        set(sprokit_test_environment
          "${prop_value}")
      else ()
        set(property "${prop}" "${prop_value}")
        list(APPEND properties
          "${property}")
      endif ()
    endif ()
  endforeach ()
endfunction ()

add_subdirectory(test)

add_subdirectory(pipeline)
add_subdirectory(pipeline_util)
