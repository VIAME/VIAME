# Fixes functions for the sprokit project
# The following functions are defined:
#
#   sprokit_use_fixes
#
# Their syntax is:
#
#   sprokit_use_fixes([name ...])
#     Set up future targets to use fixes for Boost used in sprokit.
#     Does not affect parent directories.

set(SPROKIT_FIXES "@sprokit_fixes@"
  CACHE INTERNAL "Fixes available from sprokit")

function (sprokit_use_fixes)
  foreach (fix IN LISTS ARGN)
    foreach (includedir IN LISTS SPROKIT_INCLUDE_DIR)
      include_directories(BEFORE SYSTEM "${includedir}/fixes/${fix}")
    endforeach ()
  endforeach ()
endfunction ()
