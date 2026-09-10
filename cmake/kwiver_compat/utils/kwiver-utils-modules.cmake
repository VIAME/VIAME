#
# Utility macros and functions dealing with KWIVER modules
#
# TODO: Document what this is for. It does not look to be currently utilized
#       anywhere.
#
include(CMakeParseArguments)

define_property(GLOBAL PROPERTY kwiver_modules_enabled
  BRIEF_DOCS "Active KWIVER modules"
  FULL_DOCS "List of all modules marked enabled (turned on). These are the modules to be built."
  )


#
# Add a directory as a named module.
#
#   kwiver_add_module(name directory [OPTIONAL]
#                    [DEPENDS module1 [module2 ...]])
#
# Register a module to be built. This module must be named in order to
# associate references. Names must be unique across added modules. A module
# may be labeled as OPTIONAL, creating an ENABLE/DISABLE flag for the module
# (disabled by default). Modules may also depend on other modules. This does
# not affect build dependencies, but ensures that the listed modules are
# enabled.
#
# The intended purpose for modules is ... (what is a module? How does
# it differ from a library?) Would this be like adding arrows? or arrows/core?
#
#
function(kwiver_add_module name directory)
  # Parsing args
  set(options OPTIONAL)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(module "${options}" "" "${multiValueArgs}" ${ARGN})

  if(NOT "${module_UNPARSED_ARGUMENTS}" STREQUAL "")
    message(WARNING "kwiver_add_module called with unparsed args! (${module_UNPARSED_ARGUMENTS})")
  endif()

  if(module_OPTIONAL)
    option(ENABLE_MODULE_${name} "Enable optional module ${name}" OFF)
  else()
    # else this isn't an optional module, so turn it on!
    set(ENABLE_MODULE_${name} ON)
  endif()

  if(ENABLE_MODULE_${name})
    get_property(current_modules GLOBAL PROPERTY kwiver_modules_enabled)

    # Check that name is unique among modules registered so far
    list(FIND current_modules "${name}" duplicate_index)
    if(NOT duplicate_index EQUAL -1)
      message(FATAL_ERROR "Attempted to register duplicate module '${name}'!")
    endif()

    # Check that each dep is in module enable list
    foreach(dep IN LISTS module_DEPENDS)
      list(FIND current_modules ${dep} dep_index)
      if(dep_index EQUAL -1)
        message(SEND_ERROR "Module '${name}' missing dependency: '${dep}'")
        set(module_error TRUE)
      endif()
    endforeach()
    # suppresses an optional module's enable flag from appearing if we failed above check
    if(module_error)
      return()
    endif()

    add_subdirectory("${directory}")
    kwiver_create_doxygen("${name}" "${directory}" ${module_DEPENDS})
    set_property(GLOBAL APPEND PROPERTY kwiver_modules_enabled ${name})
  endif()
endfunction()
