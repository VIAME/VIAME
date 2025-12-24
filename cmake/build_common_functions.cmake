# VIAME CMake Windows Helper
# Converts CACHE variables from build_cmake_*.cmake files to OPTIONS list format
#
# Usage in Windows platform files:
#   include(build_cmake_windows_helper.cmake)
#   include_cmake_preset(build_cmake_base.cmake)
#   include_cmake_preset(build_cmake_desktop.cmake)
#   # Add platform-specific options
#   add_option("VIAME_BUILD_KWIVER_DIR" "C:/tmp/kv1")
#   finalize_options()
#   # OPTIONS variable is now populated

# Initialize the options list
set(_VIAME_OPTIONS_LIST "")

# Macro to add an option to the list
macro(add_option VAR_NAME VAR_VALUE)
  list(APPEND _VIAME_OPTIONS_LIST "-D${VAR_NAME}=${VAR_VALUE}")
endmacro()

# Macro to add an option only if not already defined
macro(add_option_if_not_set VAR_NAME VAR_VALUE)
  list(FIND _VIAME_OPTIONS_LIST "-D${VAR_NAME}=*" _idx)
  if(_idx EQUAL -1)
    # Check if variable name appears in any existing option
    set(_found FALSE)
    foreach(_opt ${_VIAME_OPTIONS_LIST})
      if("${_opt}" MATCHES "^-D${VAR_NAME}=")
        set(_found TRUE)
        break()
      endif()
    endforeach()
    if(NOT _found)
      list(APPEND _VIAME_OPTIONS_LIST "-D${VAR_NAME}=${VAR_VALUE}")
    endif()
  endif()
endmacro()

# Override the set() command to capture CACHE variables
# This is called when including build_cmake_*.cmake files
macro(viame_capture_cache_var VAR_NAME VAR_VALUE)
  # Add to options list in CMake command-line format
  add_option_if_not_set("${VAR_NAME}" "${VAR_VALUE}")
endmacro()

# Include a cmake preset file and extract its cache variables
macro(include_cmake_preset PRESET_FILE)
  # Get the directory of this helper file
  get_filename_component(_helper_dir "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)

  # Read the preset file
  file(READ "${_helper_dir}/${PRESET_FILE}" _preset_content)

  # Parse set() commands for CACHE variables
  # Match: set(VAR_NAME "value" CACHE TYPE "description")
  # or: set(VAR_NAME value CACHE TYPE "description")
  string(REGEX MATCHALL "set\\([A-Za-z_][A-Za-z0-9_-]* [^\n]+ CACHE [^\n]+\\)" _set_commands "${_preset_content}")

  foreach(_cmd ${_set_commands})
    # Extract variable name and value
    # Handle both quoted and unquoted values
    if("${_cmd}" MATCHES "set\\(([A-Za-z_][A-Za-z0-9_-]*) \"([^\"]*)\" CACHE")
      set(_var_name "${CMAKE_MATCH_1}")
      set(_var_value "${CMAKE_MATCH_2}")
      viame_capture_cache_var("${_var_name}" "${_var_value}")
    elseif("${_cmd}" MATCHES "set\\(([A-Za-z_][A-Za-z0-9_-]*) ([A-Za-z0-9_./-]+) CACHE")
      set(_var_name "${CMAKE_MATCH_1}")
      set(_var_value "${CMAKE_MATCH_2}")
      viame_capture_cache_var("${_var_name}" "${_var_value}")
    endif()
  endforeach()
endmacro()

# Finalize and set the OPTIONS variable
macro(finalize_options)
  set(OPTIONS ${_VIAME_OPTIONS_LIST})
endmacro()
