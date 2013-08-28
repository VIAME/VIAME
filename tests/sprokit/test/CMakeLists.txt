project(sprokit_tests_test)

set(test_libraries
  sprokit_pipeline)

sprokit_discover_tests(test test_libraries test_test.cxx)

set(test_libraries
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${CMAKE_DL_LIBS})

sprokit_build_tooled_test(dlopen test_libraries test_dlopen.cxx)

foreach (library IN LISTS sprokit_libraries)
  sprokit_add_tooled_test(dlopen ${library}
    "$<TARGET_FILE:${library}>")
endforeach ()
