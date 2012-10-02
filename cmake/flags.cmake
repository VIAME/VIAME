include(CheckCXXCompilerFlag)

function (vistk_check_compiler_flag variable flag)
  string(REPLACE "+" "plus" safeflag "${flag}")

  check_cxx_compiler_flag("${flag}" "have_compiler_flag-${safeflag}")

  if ("have_compiler_flag-${safeflag}")
    if (${variable})
      set(${variable}
        "${${variable}} ${flag}"
        PARENT_SCOPE)
    else ()
      set(${variable}
        "${flag}"
        PARENT_SCOPE)
    endif ()
  endif ()
endfunction ()

set(vistk_warnings
  "")

if (MSVC)
  option(VISTK_ENABLE_DLLHELL_WARNINGS "Enables warnings about DLL visibility" OFF)
  if (NOT VISTK_ENABLE_DLLHELL_WARNINGS)
    # C4251: STL interface warnings
    vistk_check_compiler_flag(vistk_warnings /wd4251)
    # C4275: Inheritance warnings
    vistk_check_compiler_flag(vistk_warnings /wd4275)
  endif ()

  # -----------------------------------------------------------------------------
  # Disable deprecation warnings for standard C and STL functions in VS2005 and
  # later
  # -----------------------------------------------------------------------------
  if (MSVC_VERSION GREATER 1400 OR
      MSVC_VERSION EQUAL 1400)
    add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SCL_SECURE_NO_DEPRECATE)
  endif ()

  # Prevent namespace pollution.
  add_definitions(-DWIN32_LEAN_AND_MEAN)
# Assume GCC-compatible
else ()
  set(vistk_using_clang FALSE)

  check_cxx_compiler_flag(-std=c++11 has_cxx11_flags)

  cmake_dependent_option(VISTK_ENABLE_CXX11 "Enable compilation with C++11 support" OFF
    has_cxx11_flags OFF)

  # Check for clang
  if (CMAKE_CXX_COMPILER MATCHES "clang\\+\\+")
    set(vistk_using_clang TRUE)
  else ()
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}"
              -dumpversion
      WORKING_DIRECTORY
              "${vistk_source_dir}"
      RESULT_VARIABLE
              gcc_return
      OUTPUT_VARIABLE
              gcc_version)

    if (gcc_version VERSION_GREATER "4.7.0" AND
        gcc_version VERSION_LESS "4.7.2")
      if (VISTK_ENABLE_CXX11)
        message(ERROR
          "C++11 ABI is broken with GCC 4.7.0 and 4.7.1."
          "Refusing to enable C++11 support.")
      endif ()
    endif ()
  endif ()

  # Check for GCC-compatible visibility settings
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-fvisibility=hidden VISTK_HAVE_GCC_VISIBILITY)

  # Hide symbols by default
  vistk_check_compiler_flag(vistk_warnings -fvisibility=hidden)
  # Set the standard to C++98
  if (VISTK_ENABLE_CXX11)
    vistk_check_compiler_flag(vistk_warnings -std=c++11)
  else ()
    vistk_check_compiler_flag(vistk_warnings -std=c++98)
  endif ()
  # General warnings
  vistk_check_compiler_flag(vistk_warnings -Wall)
  vistk_check_compiler_flag(vistk_warnings -Wextra)
  # Class warnings
  vistk_check_compiler_flag(vistk_warnings -Wabi)
  vistk_check_compiler_flag(vistk_warnings -Wctor-dtor-privacy)
  vistk_check_compiler_flag(vistk_warnings -Winit-self)
  vistk_check_compiler_flag(vistk_warnings -Woverloaded-virtual)
  # Pointer warnings
  vistk_check_compiler_flag(vistk_warnings -Wpointer-arith)
  vistk_check_compiler_flag(vistk_warnings -Wstrict-null-sentinel)
  # Enumeration warnings
  vistk_check_compiler_flag(vistk_warnings -Wswitch-default)
  vistk_check_compiler_flag(vistk_warnings -Wswitch-enum)
  # Formatting warnings
  vistk_check_compiler_flag(vistk_warnings -Wformat-security)
  vistk_check_compiler_flag(vistk_warnings -Wformat=2)
  # Casting warnings
  vistk_check_compiler_flag(vistk_warnings -Wcast-align)
  vistk_check_compiler_flag(vistk_warnings -Wcast-qual)
  vistk_check_compiler_flag(vistk_warnings -Wdouble-promotion)
  vistk_check_compiler_flag(vistk_warnings -Wfloat-equal)
  vistk_check_compiler_flag(vistk_warnings -fstrict-overflow)
  vistk_check_compiler_flag(vistk_warnings -Wstrict-overflow=5)
  # TODO: Python triggers warnings with this
  #vistk_check_compiler_flag(vistk_warnings -Wold-style-cast)
  # Variable naming warnings
  vistk_check_compiler_flag(vistk_warnings -Wshadow)
  # C++ 11 compatibility warnings
  vistk_check_compiler_flag(vistk_warnings -Wnarrowing)
  # Exception warnings
  vistk_check_compiler_flag(vistk_warnings -Wnoexcept)
  # Miscellaneous warnings
  vistk_check_compiler_flag(vistk_warnings -Wlogical-op)
  vistk_check_compiler_flag(vistk_warnings -Wmissing-braces)
  vistk_check_compiler_flag(vistk_warnings -Wimplicit-fallthrough)

  option(VISTK_ENABLE_NITPICK "Generate warnings about nitpicky things" OFF)
  if (VISTK_ENABLE_NITPICK)
    vistk_check_compiler_flag(vistk_warnings -Wunsafe-loop-optimizations)
    vistk_check_compiler_flag(vistk_warnings -Wsign-promo)
    vistk_check_compiler_flag(vistk_warnings -Winline)
    vistk_check_compiler_flag(vistk_warnings -Weffc++)
  endif ()

  option(VISTK_ENABLE_PEDANTIC "Be pedantic" OFF)
  cmake_dependent_option(VISTK_ENABLE_PEDANTIC_ERRORS "Be REALLY pedantic" OFF
    VISTK_ENABLE_PEDANTIC OFF)
  if (VISTK_ENABLE_PEDANTIC)
    if (VISTK_ENABLE_PEDANTIC_ERRORS)
      vistk_check_compiler_flag(vistk_warnings -pedantic-errors)
    else ()
      vistk_check_compiler_flag(vistk_warnings -pedantic)
    endif ()
  endif ()

  option(VISTK_ENABLE_WERROR "Treat all warnings as errors" OFF)
  if (VISTK_ENABLE_WERROR)
    vistk_check_compiler_flag(vistk_warnings -Werror)
  endif ()

  cmake_dependent_option(VISTK_ENABLE_CLANG_CATCH_UNDEFINED_BEHAVIOR "Use clang to flag undefined behavior" OFF
    vistk_using_clang OFF)
  if (VISTK_ENABLE_CLANG_CATCH_UNDEFINED_BEHAVIOR)
    vistk_check_compiler_flag(vistk_warnings -fcatch-undefined-behavior)
  endif ()

  option(VISTK_ENABLE_COVERAGE "Build with coverage testing" OFF)
  if (VISTK_ENABLE_COVERAGE)
    set(vistk_coverage
      "")

    vistk_check_compiler_flag(vistk_coverage -O0)
    vistk_check_compiler_flag(vistk_coverage -pg)
    vistk_check_compiler_flag(vistk_coverage -fprofile-arcs)
    vistk_check_compiler_flag(vistk_coverage -ftest-coverage)

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${vistk_coverage}")
  endif ()
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${vistk_warnings}")
