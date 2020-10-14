# script to mimic the effects of activating a python virtualenv
# handles two flavors of python virtualenv
# Anaconda and Python
function (ACTIVATE_VENV path_to_venv)
  if (CONDA)
      # conda venv path stuff here, may need to actually call conda things
      # in the background and grab the output/info because of conda path
      # prefixing and the many different possible conda versions
      set (ENV{OLD_PATH} $ENV{PATH})
      if (WIN32)
        configure_file(
          ${KWIVER_CMAKE_DIR}/activate_venv.bat.in
          ${KWIVER_BINARY_DIR}/activate_venv.bat
          @ONLY
        )
        execute_process(COMMAND "cmd" "/c" "${KWIVER_BINARY_DIR}/activate_venv.bat")
      else()
        configure_file(
          ${KWIVER_CMAKE_DIR}/activate_venv.sh.in
          ${KWIVER_BINARY_DIR}/activate_venv.sh
          @ONLY
        )
        execute_process(COMMAND "bash" "-c" "${KWIVER_BINARY_DIR}/activate_venv.sh")

      endif()
      file(STRINGS "conda_venv_path.txt" tmppath)
      set(ENV{PATH} "${tmppath}:$ENV{PATH}")
      file(STRINGS "conda_prefix_path.txt" conda_prefix)
      set(ENV{CONDA_PREFIX} "${conda_prefix}")

  else()
    set (ENV{OLD_PATH} $ENV{PATH})
    if (WIN32)
        set (ENV{PATH} "${path_to_venv}/bin;$ENV{PATH}")
    else()
        set (ENV{PATH} "${path_to_venv}/bin:$ENV{PATH}")
    endif()
  endif()
endfunction()

function(DEACTIVATE_VENV)
    set (ENV{PATH} $ENV{OLD_PATH})
    unset (ENV{OLD_PATH})
    unset (ENV{VIRTUAL_ENV})
endfunction()
