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
