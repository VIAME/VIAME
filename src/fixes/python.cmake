set(__sprokit_python_fixes
  CACHE INTERNAL "Internal list of sprokit fixes")

option(SPROKIT_ENABLE_BOOST_PYTHON_FIXES "Enables fixes for Boost.Python bugs (recommended)" ON)
mark_as_advanced(SPROKIT_ENABLE_BOOST_PYTHON_FIXES)
if (SPROKIT_ENABLE_BOOST_PYTHON_FIXES)
  # Not fixed yet.
  #if (Boost_VERSION LESS 10XX00)
    # Override Boost's invoke.hpp
    sprokit_add_fix(threading __sprokit_python_fixes)
  #endif ()
  # Not fixed yet.
  #if (Boost_VERSION LESS 10XX00)
    # Override Boost's override.hpp
    sprokit_add_fix(exceptions_in_override __sprokit_python_fixes)
  #endif ()
endif ()

function (sprokit_use_python_fixes)
  foreach (fix IN LISTS __sprokit_python_fixes)
    sprokit_use_fix(${fix})
  endforeach ()
endfunction ()

function (sprokit_require_python_fixes target)
  foreach (fix IN LISTS __sprokit_python_fixes)
    sprokit_require_fix(${target} ${fix})
  endforeach ()
endfunction ()
