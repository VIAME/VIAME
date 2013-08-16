project(sprokit_fixes)

set(sprokit_fix_source_dir "${CMAKE_CURRENT_SOURCE_DIR}"
  CACHE INTERNAL "Source fix include directory path")
set(sprokit_fix_binary_dir "${CMAKE_CURRENT_BINARY_DIR}"
  CACHE INTERNAL "Configured fix include directory path")
set(sprokit_fixes
  CACHE INTERNAL "List of sprokit fixes available")

function (sprokit_add_fix fix var)
  sprokit_configure_directory(fixes-${fix}
    "${sprokit_fix_source_dir}/${fix}/boost"
    "${sprokit_fix_binary_dir}/${fix}/${BOOST_MANGLE_NAMESPACE}")
  set(sprokit_fixes
    ${sprokit_fixes}
    ${fix}
    CACHE INTERNAL "List of sprokit fixes available")
  set(${var}
    ${${var}}
    ${fix}
    CACHE INTERNAL "Internal list of sprokit fixes")
endfunction ()

function (sprokit_use_fix fix)
  if (NOT BOOST_MANGLE_NAMESPACE STREQUAL "boost")
    include_directories(BEFORE SYSTEM "${sprokit_fix_source_dir}/${fix}")
  endif ()
  include_directories(BEFORE SYSTEM "${sprokit_fix_binary_dir}/${fix}")
endfunction ()

function (sprokit_require_fix target fix)
  add_dependencies(${target}
    configure-fixes-${fix})
endfunction ()

if (SPROKIT_ENABLE_PYTHON)
  include("${CMAKE_CURRENT_SOURCE_DIR}/python.cmake")
endif ()
include("${CMAKE_CURRENT_SOURCE_DIR}/tools.cmake")

# Install the fixes.
function (sprokit_install_fix_directory directory)
  sprokit_install(
    DIRECTORY   "${directory}"
    DESTINATION include/sprokit/fixes
    COMPONENT   development)
endfunction ()

function (sprokit_install_fix fix)
  if (NOT BOOST_MANGLE_NAMESPACE STREQUAL "boost")
    sprokit_install_fix_directory("${sprokit_fix_source_dir}/${fix}")
  endif ()
  sprokit_install_fix_directory("${sprokit_fix_binary_dir}/${fix}")
endfunction ()

foreach (fix IN LISTS sprokit_fixes)
  sprokit_install_fix(${fix})
endforeach ()
