# Fixes functions for the sprokit project
# The following functions are defined:
#
#   sprokit_use_fixes
#
# Their syntax is:
#
#   sprokit_use_fixes([name ...])
#     Set up future targets to use fixes for Boost used in sprokit.
#     Does not affect parent directories. If called without arguments, all
#     known fixes are used.

define_property(GLOBAL
  PROPERTY   sprokit_fixes
  BRIEF_DOCS "A list of all fixes sprokit provides for various headers."
  FULL_DOCS  "Some things in sprokit require a patched header (usually for "
             "Boost) and as such, requires extra include directories for each "
             "fix. The fixes available depends on whether sprokit was built "
             "using them or not.")
set_property(GLOBAL
  PROPERTY sprokit_fixes
  "@sprokit_fixes@")

function (sprokit_use_fixes)
  if (ARGN)
    set(fixes "${ARGN}")
  else ()
    get_property(fixes GLOBAL
      PROPERTY sprokit_fixes)
  endif ()
  foreach (fix IN LISTS fixes)
    foreach (includedir IN LISTS SPROKIT_INCLUDE_DIR)
      include_directories(BEFORE SYSTEM "${includedir}/fixes/${fix}")
    endforeach ()
  endforeach ()
endfunction ()
