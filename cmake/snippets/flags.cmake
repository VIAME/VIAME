# Set up flags for the compiler.

include(CheckCXXCompilerFlag)

function (sprokit_add_flag flag)
  #if (ARGN)
  #  string(REPLACE "," "$<COMMA>" genflag "${flag}")
  #  string(REPLACE ";" "$<SEMICOLON>" genflag "${genflag}")
  #  string(REPLACE ">" "$<ANGLE-R>" genflag "${genflag}")
  #  foreach (config IN LISTS ARGN)
  #    add_compile_options("$<$<CONFIG:${config}>:${genflag}>")
  #  endforeach ()
  #else ()
  #  add_compile_options("${flag}")
  #endif ()
  # XXX(cmake): 2.8.12
  if (ARGN)
    foreach (config IN LISTS ARGN)
      set_property(GLOBAL APPEND_STRING
        PROPERTY "sprokit_flags_${config}"
        " ${flag}")
    endforeach ()
  else ()
    set_property(GLOBAL APPEND_STRING
      PROPERTY sprokit_flags
      " ${flag}")
  endif ()
endfunction ()

function (sprokit_want_compiler_flag flag)
  string(REPLACE "+" "plus" safeflag "${flag}")

  check_cxx_compiler_flag("${flag}" "have_compiler_flag-${safeflag}")

  if ("have_compiler_flag-${safeflag}")
    sprokit_add_flag("${flag}" ${ARGN})
  endif ()
endfunction ()

# XXX(cmake): 2.8.12
foreach (config IN LISTS CMAKE_CONFIGURATION_TYPES)
  set_property(GLOBAL
    PROPERTY "sprokit_flags_${config}")
endforeach ()
set_property(GLOBAL
  PROPERTY "sprokit_flags")

if (MSVC)
  include("${CMAKE_CURRENT_LIST_DIR}/flags-msvc.cmake")
else ()
  # Assume GCC-compatible
  include("${CMAKE_CURRENT_LIST_DIR}/flags-gnu.cmake")
endif ()

foreach (config IN LISTS CMAKE_CONFIGURATION_TYPES)
  get_property(sprokit_flags GLOBAL
    PROPERTY "sprokit_flags_${config}")
  string(TOUPPER "${config}" upper_config)
  set("CMAKE_CXX_FLAGS_${upper_config}"
    "${CMAKE_CXX_FLAGS_${upper_config}}${sprokit_flags}")
endforeach ()
get_property(sprokit_flags GLOBAL
  PROPERTY sprokit_flags)
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS}${sprokit_flags}")
# XXX(cmake): 2.8.12
if (CMAKE_CONFIGURATION_TYPES)
  foreach (config IN LISTS CMAKE_CONFIGURATION_TYPES)
    get_property(sprokit_flags GLOBAL
      PROPERTY "sprokit_flags_${config}")
    string(TOUPPER "${config}" upper_config)
    set("CMAKE_CXX_FLAGS_${upper_config}"
      "${CMAKE_CXX_FLAGS_${upper_config}}${sprokit_flags}")
  endforeach ()
else ()
  get_property(sprokit_flags GLOBAL
    PROPERTY "sprokit_flags_${CMAKE_BUILD_TYPE}")
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS}${sprokit_flags}")
endif ()
