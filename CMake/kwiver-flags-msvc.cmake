#
# Compiler flags specific to MSVC
#

option( ${CMAKE_PROJECT_NAME}_ENABLE_DLL_WARNINGS "Enable warnings about DLL visibility." OFF)
if (NOT ${CMAKE_PROJECT_NAME}_ENABLE_DLL_WARNINGS)
  kwiver_check_compiler_flag(/wd4251)
  kwiver_check_compiler_flag(/wd4275)
endif()

kwiver_check_compiler_flag(/W3)
kwiver_check_compiler_flag(/MP)

# Disable deprication warnings for standard C and STL functions in VS2005 and
# later.
if (MSVC_VERSION GREATER 1400 OR
    MSVC_VERSION EQUAL 1400)
  add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_DEPRECATE)
endif()

# Prevent namespace pollution
add_definitions(-DWIN32_LEAN_AND_MEAN)
add_definitions(-DNOMINMAX)
add_definitions(-DWINDOWS_EXTRA_LEAN)
if(${MSVC_VERSION} GREATER_EQUAL 1915)
  # You must acknowledge that you understand MSVC
  # resolved a byte alignment issue in this compiler.
  # We get this due to using Eigen objects and
  # allocating those objects with make_shared
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()
