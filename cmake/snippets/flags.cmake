# Set up flags for the compiler.

include(CheckCXXCompilerFlag)

function (sprokit_check_compiler_flag variable flag)
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

set(sprokit_warnings
  "")

if (MSVC)
  include("${CMAKE_CURRENT_LIST_DIR}/flags-msvc.cmake")
else ()
  # Assume GCC-compatible
  include("${CMAKE_CURRENT_LIST_DIR}/flags-gnu.cmake")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${sprokit_warnings}")
