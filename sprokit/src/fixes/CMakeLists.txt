project(sprokit_fixes)

set_property(GLOBAL
  PROPERTY sprokit_fix_source_dir
  "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(GLOBAL
  PROPERTY sprokit_fix_binary_dir
  "${CMAKE_CURRENT_BINARY_DIR}")
set_property(GLOBAL
  PROPERTY sprokit_fixes)

function (sprokit_add_fix fix property)
  get_property(sprokit_fix_source_dir GLOBAL
    PROPERTY sprokit_fix_source_dir)
  get_property(sprokit_fix_binary_dir GLOBAL
    PROPERTY sprokit_fix_binary_dir)
  sprokit_configure_directory("fixes-${fix}"
    "${sprokit_fix_source_dir}/${fix}/boost"
    "${sprokit_fix_binary_dir}/${fix}/${BOOST_MANGLE_NAMESPACE}")
  set_property(GLOBAL APPEND
    PROPERTY sprokit_fixes
    "${fix}")
  set_property(GLOBAL APPEND
    PROPERTY "${property}"
    "${fix}")
endfunction ()

function (sprokit_use_fix fix)
  if (NOT BOOST_MANGLE_NAMESPACE STREQUAL "boost")
    get_property(sprokit_fix_source_dir GLOBAL
      PROPERTY sprokit_fix_source_dir)
    include_directories(BEFORE SYSTEM "${sprokit_fix_source_dir}/${fix}")
  endif ()
  get_property(sprokit_fix_binary_dir GLOBAL
    PROPERTY sprokit_fix_binary_dir)
  include_directories(BEFORE SYSTEM "${sprokit_fix_binary_dir}/${fix}")
endfunction ()

function (sprokit_require_fix target fix)
  add_dependencies("${target}"
    "configure-fixes-${fix}")
endfunction ()

if (KWIVER_ENABLE_PYTHON)
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
    get_property(sprokit_fix_source_dir GLOBAL
      PROPERTY sprokit_fix_source_dir)
    sprokit_install_fix_directory("${sprokit_fix_source_dir}/${fix}")
  endif ()
  get_property(sprokit_fix_binary_dir GLOBAL
    PROPERTY sprokit_fix_binary_dir)
  sprokit_install_fix_directory("${sprokit_fix_binary_dir}/${fix}")
endfunction ()

get_property(fixes GLOBAL
  PROPERTY sprokit_fixes)
foreach (fix IN LISTS fixes)
  sprokit_install_fix("${fix}")
endforeach ()
