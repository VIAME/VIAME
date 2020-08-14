#
# Compiler flags specific to use with GCC
#

include( CMakeDependentOption )

kwiver_check_compiler_flag( -std=c++11 -std=c++0x )
kwiver_check_compiler_flag( -pthread )
kwiver_check_compiler_flag( -fvisibility=hidden )
kwiver_check_compiler_flag( -Wall )
kwiver_check_compiler_flag( -Werror=return-type )
kwiver_check_compiler_flag( -Werror=non-virtual-dtor )
kwiver_check_compiler_flag( -Werror=narrowing )
kwiver_check_compiler_flag( -Werror=init-self )
kwiver_check_compiler_flag( -Werror=reorder )
kwiver_check_compiler_flag( -Werror=overloaded-virtual )
kwiver_check_compiler_flag( -Werror=cast-qual )
kwiver_check_compiler_flag( -Werror=vla )

# to slience this warning
kwiver_check_compiler_flag( -Wno-unknown-pragmas )

# linker shared object control flags
kwiver_check_compiler_flag( -Wl,--no-undefined )
kwiver_check_compiler_flag( -Wl,--copy-dt-needed-entries )


OPTION(KWIVER_CPP_EXTRA "Generate more warnings about bad practices" OFF)
mark_as_advanced( KWIVER_CPP_EXTRA)
if (KWIVER_CPP_EXTRA)
  # General warnings
  kwiver_check_compiler_flag(-Wextra)
  # Class warnings
  kwiver_check_compiler_flag(-Wabi)
  kwiver_check_compiler_flag(-Wctor-dtor-privacy)
  kwiver_check_compiler_flag(-Winit-self)
  kwiver_check_compiler_flag(-Woverloaded-virtual)
  # Pointer warnings
  kwiver_check_compiler_flag(-Wpointer-arith)
  kwiver_check_compiler_flag(-Wstrict-null-sentinel)
  # Enumeration warnings
  kwiver_check_compiler_flag(-Wswitch-default)
  kwiver_check_compiler_flag(-Wswitch-enum)
  # Formatting warnings
  kwiver_check_compiler_flag(-Wformat-security)
  kwiver_check_compiler_flag(-Wformat=2)
  # Casting warnings
  kwiver_check_compiler_flag(-Wcast-align)
  kwiver_check_compiler_flag(-Wcast-qual)
  kwiver_check_compiler_flag(-Wdouble-promotion)
  kwiver_check_compiler_flag(-Wfloat-equal)
  kwiver_check_compiler_flag(-fstrict-overflow)
  kwiver_check_compiler_flag(-Wstrict-overflow=5)

  # TODO: Python triggers warnings with this
  kwiver_check_compiler_flag(-Wold-style-cast)
  # Variable naming warnings
  kwiver_check_compiler_flag(-Wshadow)
  # Exception warnings
  kwiver_check_compiler_flag(-Wnoexcept)
  # Miscellaneous warnings
  kwiver_check_compiler_flag(-Wlogical-op)
  kwiver_check_compiler_flag(-Wmissing-braces)
  kwiver_check_compiler_flag(-Wimplicit-fallthrough)
  kwiver_check_compiler_flag(-Wdocumentation)
  kwiver_check_compiler_flag(-Wundef)
  kwiver_check_compiler_flag(-Wunused-macros)
endif()

OPTION(KWIVER_CPP_NITPICK "Generate warnings about nitpicky things" OFF)
mark_as_advanced(KWIVER_CPP_NITPICK)
if (KWIVER_CPP_NITPICK)
  kwiver_check_compiler_flag(-Wunsafe-loop-optimizations)
  kwiver_check_compiler_flag(-Wsign-promo)
  kwiver_check_compiler_flag(-Winline)
  kwiver_check_compiler_flag(-Weffc++)
endif ()

option(KWIVER_CPP_PEDANTIC "Be pedantic" OFF)
mark_as_advanced(KWIVER_CPP_PEDANTIC)
cmake_dependent_option(KWIVER_CPP_PEDANTIC_ERRORS "Be REALLY pedantic" OFF
  KWIVER_CPP_PEDANTIC OFF)
if (KWIVER_CPP_PEDANTIC)
  if (KWIVER_CPP_PEDANTIC_ERRORS)
    kwiver_check_compiler_flag(-pedantic-errors)
  else ()
    kwiver_check_compiler_flag(-pedantic)
  endif ()
endif ()

OPTION(KWIVER_CPP_WERROR "Treat all warnings as errors" OFF)
if (KWIVER_CPP_WERROR)
  kwiver_check_compiler_flag(-Werror)
endif ()

CMAKE_DEPENDENT_OPTION(KWIVER_CPP_CLANG_CATCH_UNDEFINED_BEHAVIOR "Use clang to flag undefined behavior" OFF
  kwiver_using_clang OFF)
if (KWIVER_CPP_CLANG_CATCH_UNDEFINED_BEHAVIOR)
  kwiver_check_compiler_flag(-fcatch-undefined-behavior)
endif ()

OPTION(KWIVER_CPP_ASAN "Enable address sanitization" OFF)
mark_as_advanced(KWIVER_CPP_ASAN)
if (KWIVER_CPP_ASAN)
  kwiver_check_compiler_flag(-fsanitize=address)
  kwiver_check_compiler_flag(-fno-omit-frame-pointer)
endif ()

## only in debug mode/config
OPTION(KWIVER_CPP_COVERAGE "Build with coverage testing" OFF)
mark_as_advanced(KWIVER_CPP_COVERAGE)
if (KWIVER_CPP_COVERAGE   AND   CMAKE_BUILD_TYPE EQUAL "DEBUG")
  kwiver_check_compiler_flag(-O0)
  kwiver_check_compiler_flag(-pg)
  kwiver_check_compiler_flag(-ftest-coverage)
  # It seems as though the flag isn't detected alone.
  kwiver_check_compiler_flag(-fprofile-arcs)
endif ()

# GCC Flag used for stripping binaries of any unused symbols
# Used for reducing the size of kwiver wheel since pypi has 60Mb constraint on wheel size
if (SKBUILD)
  kwiver_check_compiler_flag( -s )
endif ()
