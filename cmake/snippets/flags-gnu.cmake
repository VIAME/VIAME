set(sprokit_using_clang FALSE)

check_cxx_compiler_flag(-std=c++11 has_cxx11_flags)

cmake_dependent_option(SPROKIT_ENABLE_CXX11 "Enable compilation with C++11 support" OFF
  has_cxx11_flags OFF)

# Check for clang
if (CMAKE_CXX_COMPILER MATCHES "clang\\+\\+")
  set(sprokit_using_clang TRUE)
else ()
  # XXX(gcc): 4.7.2
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
sprokit_want_compiler_flag(-fvisibility=hidden)
# Set the standard to C++98
if (SPROKIT_ENABLE_CXX11)
  sprokit_want_compiler_flag(-std=c++11)
else ()
  sprokit_want_compiler_flag(-std=c++98)
endif ()
# General warnings
sprokit_want_compiler_flag(-Wall)
sprokit_want_compiler_flag(-Wextra)
# Class warnings
sprokit_want_compiler_flag(-Wabi)
sprokit_want_compiler_flag(-Wctor-dtor-privacy)
sprokit_want_compiler_flag(-Winit-self)
sprokit_want_compiler_flag(-Woverloaded-virtual)
# Pointer warnings
sprokit_want_compiler_flag(-Wpointer-arith)
sprokit_want_compiler_flag(-Wstrict-null-sentinel)
# Enumeration warnings
sprokit_want_compiler_flag(-Wswitch-default)
sprokit_want_compiler_flag(-Wswitch-enum)
# Formatting warnings
sprokit_want_compiler_flag(-Wformat-security)
sprokit_want_compiler_flag(-Wformat=2)
# Casting warnings
sprokit_want_compiler_flag(-Wcast-align)
sprokit_want_compiler_flag(-Wcast-qual)
sprokit_want_compiler_flag(-Wdouble-promotion)
sprokit_want_compiler_flag(-Wfloat-equal)
sprokit_want_compiler_flag(-fstrict-overflow)
sprokit_want_compiler_flag(-Wstrict-overflow=5)
# TODO: Python triggers warnings with this
#sprokit_want_compiler_flag(-Wold-style-cast)
# Variable naming warnings
sprokit_want_compiler_flag(-Wshadow)
# C++ 11 compatibility warnings
sprokit_want_compiler_flag(-Wnarrowing)
# Exception warnings
sprokit_want_compiler_flag(-Wnoexcept)
# Miscellaneous warnings
sprokit_want_compiler_flag(-Wlogical-op)
sprokit_want_compiler_flag(-Wmissing-braces)
sprokit_want_compiler_flag(-Wimplicit-fallthrough)
sprokit_want_compiler_flag(-Wdocumentation)
sprokit_want_compiler_flag(-Wundef)
sprokit_want_compiler_flag(-Wunused-macros)

option(SPROKIT_ENABLE_NITPICK "Generate warnings about nitpicky things" OFF)
if (SPROKIT_ENABLE_NITPICK)
  sprokit_want_compiler_flag(-Wunsafe-loop-optimizations)
  sprokit_want_compiler_flag(-Wsign-promo)
  sprokit_want_compiler_flag(-Winline)
  sprokit_want_compiler_flag(-Weffc++)
endif ()

option(SPROKIT_ENABLE_PEDANTIC "Be pedantic" OFF)
cmake_dependent_option(SPROKIT_ENABLE_PEDANTIC_ERRORS "Be REALLY pedantic" OFF
  SPROKIT_ENABLE_PEDANTIC OFF)
if (SPROKIT_ENABLE_PEDANTIC)
  if (SPROKIT_ENABLE_PEDANTIC_ERRORS)
    sprokit_want_compiler_flag(-pedantic-errors)
  else ()
    sprokit_want_compiler_flag(-pedantic)
  endif ()
endif ()

option(SPROKIT_ENABLE_WERROR "Treat all warnings as errors" OFF)
if (SPROKIT_ENABLE_WERROR)
  sprokit_want_compiler_flag(-Werror)
endif ()

cmake_dependent_option(SPROKIT_ENABLE_CLANG_CATCH_UNDEFINED_BEHAVIOR "Use clang to flag undefined behavior" OFF
  sprokit_using_clang OFF)
if (SPROKIT_ENABLE_CLANG_CATCH_UNDEFINED_BEHAVIOR)
  sprokit_want_compiler_flag(-fcatch-undefined-behavior)
endif ()

option(SPROKIT_ENABLE_ASAN "Enable address sanitization" OFF)
if (SPROKIT_ENABLE_ASAN)
  sprokit_check_compiler_flag(sprokit_warnings -fsanitize=address)
  sprokit_check_compiler_flag(sprokit_warnings -fno-omit-frame-pointer)
endif ()

option(SPROKIT_ENABLE_COVERAGE "Build with coverage testing" OFF)
if (SPROKIT_ENABLE_COVERAGE)
  sprokit_want_compiler_flag(-O0 Debug)
  sprokit_want_compiler_flag(-pg Debug)
  sprokit_want_compiler_flag(-ftest-coverage Debug)
  # It seems as though the flag isn't detected alone.
  sprokit_add_flag(-fprofile-arcs Debug)
endif ()
