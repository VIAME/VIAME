# kwiver_package_option(<PACKAGE_NAME>
#   [DESCRIPTION <description>]
#   [FLETCH_NAME <name>]
#   [DEFAULT <value>]
# )
#
# Declares option KWIVER_ENABLE_<PACKAGE_NAME> with a default resolved in order:
#   1. fletch_ENABLED_<FLETCH_NAME> if defined
#   2. KWIVER_ENABLE_<PACKAGE_NAME> if already set in cache
#   3. DEFAULT (defaults to OFF)
#
# Also sets kwiver_enabled_<package_name> (lowercase) in the calling scope.
# FLETCH_NAME defaults to <PACKAGE_NAME> when not specified.

function(kwiver_package_option _pkg)
  cmake_parse_arguments(KPO "" "DESCRIPTION;FLETCH_NAME;DEFAULT" "" ${ARGN})

  if(NOT DEFINED KPO_DESCRIPTION)
    set(KPO_DESCRIPTION "Enable ${_pkg} dependent code and plugins")
  endif()
  if(NOT DEFINED KPO_FLETCH_NAME)
    set(KPO_FLETCH_NAME ${_pkg})
  endif()
  if(NOT DEFINED KPO_DEFAULT)
    set(KPO_DEFAULT OFF)
  endif()

  if(DEFINED fletch_ENABLED_${KPO_FLETCH_NAME})
    set(_default ${fletch_ENABLED_${KPO_FLETCH_NAME}})
  elseif(DEFINED KWIVER_ENABLE_${_pkg})
    set(_default ${KWIVER_ENABLE_${_pkg}})
  else()
    set(_default ${KPO_DEFAULT})
  endif()

  option(KWIVER_ENABLE_${_pkg}
    "${KPO_DESCRIPTION}"
    ${_default}
  )

  string(TOLOWER "${_pkg}" _pkg_lower)
  set(kwiver_enabled_${_pkg_lower} ${KWIVER_ENABLE_${_pkg}} PARENT_SCOPE)
endfunction()
