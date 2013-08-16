set(sprokit_using_clang FALSE)

check_cxx_compiler_flag(-std=c++11 has_cxx11_flags)

cmake_dependent_option(SPROKIT_ENABLE_CXX11 "Enable compilation with C++11 support" OFF
  has_cxx11_flags OFF)

# Check for clang
if (CMAKE_CXX_COMPILER MATCHES "clang\\+\\+")
  set(sprokit_using_clang TRUE)
else ()
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.7.0" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.7.2")
    if (SPROKIT_ENABLE_CXX11)
      message(SEND_ERROR
        "C++11 ABI is broken with GCC 4.7.0 and 4.7.1."
        "Refusing to enable C++11 support.")
    endif ()
  endif ()
endif ()

# Check for GCC-compatible visibility settings
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fvisibility=hidden SPROKIT_HAVE_GCC_VISIBILITY)

# Hide symbols by default
sprokit_check_compiler_flag(sprokit_warnings -fvisibility=hidden)
# Set the standard to C++98
if (SPROKIT_ENABLE_CXX11)
  sprokit_check_compiler_flag(sprokit_warnings -std=c++11)
else ()
  sprokit_check_compiler_flag(sprokit_warnings -std=c++98)
endif ()
# General warnings
sprokit_check_compiler_flag(sprokit_warnings -Wall)
sprokit_check_compiler_flag(sprokit_warnings -Wextra)
# Class warnings
sprokit_check_compiler_flag(sprokit_warnings -Wabi)
sprokit_check_compiler_flag(sprokit_warnings -Wctor-dtor-privacy)
sprokit_check_compiler_flag(sprokit_warnings -Winit-self)
sprokit_check_compiler_flag(sprokit_warnings -Woverloaded-virtual)
# Pointer warnings
sprokit_check_compiler_flag(sprokit_warnings -Wpointer-arith)
sprokit_check_compiler_flag(sprokit_warnings -Wstrict-null-sentinel)
# Enumeration warnings
sprokit_check_compiler_flag(sprokit_warnings -Wswitch-default)
sprokit_check_compiler_flag(sprokit_warnings -Wswitch-enum)
# Formatting warnings
sprokit_check_compiler_flag(sprokit_warnings -Wformat-security)
sprokit_check_compiler_flag(sprokit_warnings -Wformat=2)
# Casting warnings
sprokit_check_compiler_flag(sprokit_warnings -Wcast-align)
sprokit_check_compiler_flag(sprokit_warnings -Wcast-qual)
sprokit_check_compiler_flag(sprokit_warnings -Wdouble-promotion)
sprokit_check_compiler_flag(sprokit_warnings -Wfloat-equal)
sprokit_check_compiler_flag(sprokit_warnings -fstrict-overflow)
sprokit_check_compiler_flag(sprokit_warnings -Wstrict-overflow=5)
# TODO: Python triggers warnings with this
#sprokit_check_compiler_flag(sprokit_warnings -Wold-style-cast)
# Variable naming warnings
sprokit_check_compiler_flag(sprokit_warnings -Wshadow)
# C++ 11 compatibility warnings
sprokit_check_compiler_flag(sprokit_warnings -Wnarrowing)
# Exception warnings
sprokit_check_compiler_flag(sprokit_warnings -Wnoexcept)
# Miscellaneous warnings
sprokit_check_compiler_flag(sprokit_warnings -Wlogical-op)
sprokit_check_compiler_flag(sprokit_warnings -Wmissing-braces)
sprokit_check_compiler_flag(sprokit_warnings -Wimplicit-fallthrough)

option(SPROKIT_ENABLE_NITPICK "Generate warnings about nitpicky things" OFF)
if (SPROKIT_ENABLE_NITPICK)
  sprokit_check_compiler_flag(sprokit_warnings -Wunsafe-loop-optimizations)
  sprokit_check_compiler_flag(sprokit_warnings -Wsign-promo)
  sprokit_check_compiler_flag(sprokit_warnings -Winline)
  sprokit_check_compiler_flag(sprokit_warnings -Weffc++)
endif ()

option(SPROKIT_ENABLE_PEDANTIC "Be pedantic" OFF)
cmake_dependent_option(SPROKIT_ENABLE_PEDANTIC_ERRORS "Be REALLY pedantic" OFF
  SPROKIT_ENABLE_PEDANTIC OFF)
if (SPROKIT_ENABLE_PEDANTIC)
  if (SPROKIT_ENABLE_PEDANTIC_ERRORS)
    sprokit_check_compiler_flag(sprokit_warnings -pedantic-errors)
  else ()
    sprokit_check_compiler_flag(sprokit_warnings -pedantic)
  endif ()
endif ()

option(SPROKIT_ENABLE_WERROR "Treat all warnings as errors" OFF)
if (SPROKIT_ENABLE_WERROR)
  sprokit_check_compiler_flag(sprokit_warnings -Werror)
endif ()

cmake_dependent_option(SPROKIT_ENABLE_CLANG_CATCH_UNDEFINED_BEHAVIOR "Use clang to flag undefined behavior" OFF
  sprokit_using_clang OFF)
if (SPROKIT_ENABLE_CLANG_CATCH_UNDEFINED_BEHAVIOR)
  sprokit_check_compiler_flag(sprokit_warnings -fcatch-undefined-behavior)
endif ()

option(SPROKIT_ENABLE_COVERAGE "Build with coverage testing" OFF)
if (SPROKIT_ENABLE_COVERAGE)
  set(sprokit_coverage
    "")

  sprokit_check_compiler_flag(sprokit_coverage -O0)
  sprokit_check_compiler_flag(sprokit_coverage -pg)
  sprokit_check_compiler_flag(sprokit_coverage -ftest-coverage)

  # It seems as though the flag isn't detected alone.
  set(sprokit_coverage
    "${sprokit_coverage} -fprofile-arcs")

  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${sprokit_coverage}")
endif ()
