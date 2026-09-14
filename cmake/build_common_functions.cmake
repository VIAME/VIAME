# Configure presets for the ctest dashboard scripts
#
# The Windows nightly scripts (`build_server_windows*.cmake`) drive a build
# through `ctest -S`, which configures with an OPTIONS list rather than with
# `cmake --preset`. These read the same presets `CMakePresets.json` gives
# everyone else, so there is one place a configuration is written.
#
# Usage in a dashboard script:
#   include(build_common_functions.cmake)
#   include_cmake_preset(windows-gpu)
#   # platform-specific additions
#   add_option("CUDA_NVCC_EXECUTABLE:PATH" "C:/...")
#   finalize_options()
#   # OPTIONS now holds -D entries for ctest_configure
#
# Until P10 these read `build_cmake_*.cmake` cache fragments, matching
# `set(... CACHE ...)` lines with regular expressions.

set(_VIAME_OPTIONS_LIST "")

# -D entry for a variable, replacing any entry already in the list for it.
macro(add_option VAR_NAME VAR_VALUE)
  string(REGEX REPLACE ":.*$" "" _viame_bare "${VAR_NAME}")
  set(_viame_kept "")
  foreach(_viame_opt IN LISTS _VIAME_OPTIONS_LIST)
    if(NOT "${_viame_opt}" MATCHES "^-D${_viame_bare}(:[A-Z]+)?=")
      list(APPEND _viame_kept "${_viame_opt}")
    endif()
  endforeach()
  set(_VIAME_OPTIONS_LIST ${_viame_kept})
  list(APPEND _VIAME_OPTIONS_LIST "-D${VAR_NAME}=${VAR_VALUE}")
endmacro()

# The cache variables of one configure preset, its `inherits` first -- in
# order, so a later parent overrides an earlier one, as CMake applies them --
# and then its own.
function(_viame_preset_variables presets_json preset_name out_names out_values)
  string(JSON _count LENGTH "${presets_json}" configurePresets)
  math(EXPR _last "${_count} - 1")

  set(_found FALSE)
  foreach(_i RANGE ${_last})
    string(JSON _name GET "${presets_json}" configurePresets ${_i} name)
    if(_name STREQUAL preset_name)
      set(_found TRUE)
      string(JSON _preset GET "${presets_json}" configurePresets ${_i})
      break()
    endif()
  endforeach()

  if(NOT _found)
    message(FATAL_ERROR "No configure preset named '${preset_name}' in CMakePresets.json")
  endif()

  set(_names "")
  set(_values "")

  string(JSON _inherits_type ERROR_VARIABLE _no_inherits TYPE "${_preset}" inherits)
  if(NOT _no_inherits)
    if(_inherits_type STREQUAL "STRING")
      string(JSON _parent GET "${_preset}" inherits)
      set(_parents "${_parent}")
    else()
      set(_parents "")
      string(JSON _n LENGTH "${_preset}" inherits)
      math(EXPR _n_last "${_n} - 1")
      foreach(_j RANGE ${_n_last})
        string(JSON _parent GET "${_preset}" inherits ${_j})
        list(APPEND _parents "${_parent}")
      endforeach()
    endif()

    # CMake gives the first listed parent precedence; applying the list in
    # reverse lets each earlier parent overwrite the later ones.
    list(REVERSE _parents)
    foreach(_parent IN LISTS _parents)
      _viame_preset_variables("${presets_json}" "${_parent}" _p_names _p_values)
      list(LENGTH _p_names _p_count)
      if(_p_count GREATER 0)
        math(EXPR _p_last "${_p_count} - 1")
        foreach(_k RANGE ${_p_last})
          list(GET _p_names ${_k} _v_name)
          list(GET _p_values ${_k} _v_value)
          list(FIND _names "${_v_name}" _at)
          if(_at EQUAL -1)
            list(APPEND _names "${_v_name}")
            list(APPEND _values "${_v_value}")
          else()
            list(REMOVE_AT _values ${_at})
            list(INSERT _values ${_at} "${_v_value}")
          endif()
        endforeach()
      endif()
    endforeach()
  endif()

  string(JSON _vars_type ERROR_VARIABLE _no_vars TYPE "${_preset}" cacheVariables)
  if(NOT _no_vars)
    string(JSON _v_count LENGTH "${_preset}" cacheVariables)
    if(_v_count GREATER 0)
      math(EXPR _v_last "${_v_count} - 1")
      foreach(_k RANGE ${_v_last})
        string(JSON _v_name MEMBER "${_preset}" cacheVariables ${_k})
        string(JSON _v_type TYPE "${_preset}" cacheVariables "${_v_name}")
        if(_v_type STREQUAL "OBJECT")
          string(JSON _v_value GET "${_preset}" cacheVariables "${_v_name}" value)
          string(JSON _v_cache_type ERROR_VARIABLE _no_type GET "${_preset}" cacheVariables "${_v_name}" type)
          if(NOT _no_type)
            set(_v_name "${_v_name}:${_v_cache_type}")
          endif()
        else()
          string(JSON _v_value GET "${_preset}" cacheVariables "${_v_name}")
        endif()
        if(_v_value STREQUAL "ON" OR _v_value STREQUAL "OFF")
          # already CMake's spelling
        elseif(_v_type STREQUAL "BOOLEAN")
          if(_v_value)
            set(_v_value ON)
          else()
            set(_v_value OFF)
          endif()
        endif()

        string(REGEX REPLACE ":.*$" "" _bare "${_v_name}")
        set(_at -1)
        set(_idx 0)
        foreach(_existing IN LISTS _names)
          string(REGEX REPLACE ":.*$" "" _existing_bare "${_existing}")
          if(_existing_bare STREQUAL _bare)
            set(_at ${_idx})
          endif()
          math(EXPR _idx "${_idx} + 1")
        endforeach()
        if(_at EQUAL -1)
          list(APPEND _names "${_v_name}")
          list(APPEND _values "${_v_value}")
        else()
          list(REMOVE_AT _names ${_at})
          list(INSERT _names ${_at} "${_v_name}")
          list(REMOVE_AT _values ${_at})
          list(INSERT _values ${_at} "${_v_value}")
        endif()
      endforeach()
    endif()
  endif()

  set(${out_names} "${_names}" PARENT_SCOPE)
  set(${out_values} "${_values}" PARENT_SCOPE)
endfunction()

# Add every cache variable a configure preset sets, inherited ones included.
macro(include_cmake_preset PRESET_NAME)
  get_filename_component(_viame_presets_file
    "${CMAKE_CURRENT_LIST_DIR}/../CMakePresets.json" ABSOLUTE)
  file(READ "${_viame_presets_file}" _viame_presets_json)
  _viame_preset_variables("${_viame_presets_json}" "${PRESET_NAME}"
    _viame_preset_names _viame_preset_values)
  list(LENGTH _viame_preset_names _viame_preset_count)
  if(_viame_preset_count GREATER 0)
    math(EXPR _viame_preset_last "${_viame_preset_count} - 1")
    foreach(_viame_i RANGE ${_viame_preset_last})
      list(GET _viame_preset_names ${_viame_i} _viame_name)
      list(GET _viame_preset_values ${_viame_i} _viame_value)
      add_option("${_viame_name}" "${_viame_value}")
    endforeach()
  endif()
endmacro()

macro(finalize_options)
  set(OPTIONS ${_VIAME_OPTIONS_LIST})
endmacro()
