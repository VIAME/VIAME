set(__sprokit_tools_fixes
  CACHE INTERNAL "Internal list of sprokit fixes")

option(SPROKIT_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES "Enables fixes for boost::program_options bugs (recommended)" ON)
mark_as_advanced(SPROKIT_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES)
if (SPROKIT_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES)
  if (Boost_VERSION LESS 105000)
    # Override Boost's value_semantic.hpp
    sprokit_add_fix(value_name __sprokit_tools_fixes)
  endif ()
endif ()

function (sprokit_use_tools_fixes)
  foreach (fix IN LISTS __sprokit_tools_fixes)
    sprokit_use_fix(${fix})
  endforeach ()
endfunction ()

function (sprokit_require_tools_fixes target)
  foreach (fix IN LISTS __sprokit_tools_fixes)
    sprokit_require_fix(${target} ${fix})
  endforeach ()
endfunction ()
