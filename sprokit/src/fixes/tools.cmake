set_property(GLOBAL
  PROPERTY sprokit_tools_fixes)

option(SPROKIT_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES "Enables fixes for boost::program_options bugs (recommended)" ON)
mark_as_advanced(SPROKIT_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES)
if (SPROKIT_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES)
  # XXX(Boost): 1.50.0
  if (Boost_VERSION LESS 105000)
    # Override Boost's value_semantic.hpp
    sprokit_add_fix(value_name sprokit_tools_fixes)
  endif ()
endif ()

function (sprokit_use_tools_fixes)
  get_property(fixes GLOBAL
    PROPERTY sprokit_tools_fixes)
  foreach (fix IN LISTS fixes)
    sprokit_use_fix("${fix}")
  endforeach ()
endfunction ()

function (sprokit_require_tools_fixes target)
  get_property(fixes GLOBAL
    PROPERTY sprokit_tools_fixes)
  foreach (fix IN LISTS fixes)
    sprokit_require_fix("${target}" "${fix}")
  endforeach ()
endfunction ()
