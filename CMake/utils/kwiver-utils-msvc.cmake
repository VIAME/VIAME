
#define_property(GLOBAL PROPERTY kwiver_msvc_env
#  BRIEF_DOCS "Values set by setup environment batch files"
#  FULL_DOCS "All environment variables needed for running executables through msvc."
#  )

#+
# Create a variable with the windows environment for use in MSVC
#
# SETUP_BATCH_FILES is a list of batch scripts on disk to process
#
# Each batch file will be read in and each 'set' of an environment variable
# will be extracted and reformed with the proper values for MSVC
#
# This logic will ignore 'config' variables as that is generally
# used in batch files to signal the build type folder to use in batch scripts
#-


if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
  # Addresses problems with the use of IN_LIST below
endif()

function(kwiver_setup_msvc_env)
  foreach(setup_batch ${ARGN})
    list(APPEND SETUP_BATCH_FILES "${setup_batch}")
  endforeach()

  foreach(setup_batch ${SETUP_BATCH_FILES})
    #message(STATUS "Extracting environment from ${setup_batch}")
    get_filename_component(batch_dir ${setup_batch} DIRECTORY)
    file(READ "${setup_batch}" contents)
    # Convert file contents into a CMake list
    # Where each element in the list is one line of the file
    string(REGEX REPLACE ";" "\\\\;" contents "${contents}")
    string(REGEX REPLACE "\n" ";" contents "${contents}")
    foreach(line ${contents})
      string(STRIP "${line}" line)
      string(TOLOWER "${line}" search)
      # Ignore comments (:: and rem)
      string(FIND "${search}" "::" idx)
      if("${idx}" GREATER -1)
        continue()
      endif()
      string(FIND "${search}" "rem" idx)
      if("${idx}" GREATER -1)
        continue()
      endif()
      # Ignore the set config line (a kwiver idiom)
      string(FIND "${search}" "set config" idx)
      if("${idx}" GREATER -1)
        continue()
      endif()
      # Is this line setting an env variable?
      string(FIND "${search}" "set " idx)
      string(FIND "${search}" "=" edx)
      if(NOT "${idx}" EQUAL 0 OR ${edx} EQUAL -1)
        continue()
      endif()

      #message(STATUS "I cleaning ${line}")
      # The line format is
      # set SOMETHING = VALUE
      # So we need to make a 'map' of variable names and their list of values
      # Lots of string index manipulation to get the right indexes to chop of 'set' and '=' strings
      string(LENGTH "${line}" length)
      math(EXPR _vardx "${edx}-4")
      math(EXPR _valdx "${edx}+1")
      string(SUBSTRING "${line}" 4 ${_vardx} _var) # what are we setting?
      string(SUBSTRING "${line}" ${_valdx} ${length} _val) # what are we setting it to?
      # replace batch terms with msvc terms
      string(REPLACE "%config%" "$(Configuration)" _val "${_val}")
      string(REPLACE "%~dp0" "${batch_dir}" _val "${_val}")
      # check for any recursive sets
      # such as PATH=%something%;%PATH% <-- remove the %PATH%
      string(FIND "${_val}" "%${_var}%" idx)
      # remove recursive sets
      string(REPLACE "%${_var}%" "" _val "${_val}")
      #message(STATUS "I am setting ${_var} to ${_val}")
      # Keep track of setting the same variable over and over
      if(NOT ${_var} IN_LIST _env_variables)
        list(APPEND _env_variables "${_var}")
      endif()
      # if a recursive variable was found append values, else replace
      if("${idx}" GREATER -1)
        list(APPEND "_env_${_var}" "${_val}")
      else()
        set("_env_${_var}" "${_val}")
      endif()
    endforeach()
  endforeach()
  # Consolidate setting env variables into one line
  foreach(_env_var ${_env_variables})
    #message(STATUS "processing ${_env_var}")
    set(MSVC_ENV "${MSVC_ENV}${_env_var}=")
    foreach(_env_val ${_env_${_env_var}})
      set(MSVC_ENV "${MSVC_ENV}${_env_val};")
    endforeach()
    set(MSVC_ENV "${MSVC_ENV}\n")
  endforeach()
  # Remove semi-colons at the end of lines
  string(REPLACE ";\n" "\n" MSVC_ENV "${MSVC_ENV}")
  #message(STATUS "Setting MSVC environment to ${MSVC_ENV}")

  # Now loop over all the executable we made and provide a vcxproj file for them
  get_property(executables GLOBAL PROPERTY kwiver_executables)
  get_property(executables_paths GLOBAL PROPERTY kwiver_executables_paths)
  #message(STATUS "Create msvc files for these executables : ${executables}")
  # Loop over our 2 lists of the same length at the same time
  # So go from 0 to length-1
  list(LENGTH executables stop)
  math(EXPR stop "${stop}-1")
  foreach(i RANGE 0 ${stop})
    # split the exe into name and binary dir
    list(GET executables ${i} exe)
    list(GET executables_paths ${i} exe_binary_dir)

    #message(STATUS "Setting MSVC environment for ${exe} in ${exe_binary_dir}")
    configure_file(
      ${KWIVER_CMAKE_DIR}/vcxproj.user.in
      ${exe_binary_dir}/${exe}.vcxproj.user
      @ONLY
    )
  endforeach()
endfunction()
