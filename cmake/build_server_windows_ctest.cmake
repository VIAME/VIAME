# Get platform specific build info
include(build_server_windows.cmake)

# Run CTest
ctest_start(${CTEST_BUILD_MODEL})
ctest_configure(BUILD ${CTEST_BINARY_DIRECTORY} SOURCE ${CTEST_SOURCE_DIRECTORY}
                OPTIONS "${OPTIONS}")
ctest_build()
ctest_submit()
