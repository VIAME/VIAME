# script to mimic the effects of activating a python virtualenv
# mimicing the command 'source <path/to/venv/>/bin/activate'
if ( ACTIVATE)
  set (ENV{OLD_PATH} $ENV{PATH})
  set (ENV{VIRTUAL_ENV} "${KWIVER_BINARY_DIR}/testing_venv")
  if (WIN32)
    set (ENV{PATH} "$ENV{VIRTUAL_ENV}/bin;$ENV{PATH}")
  else()
    set (ENV{PATH} "${KWIVER_BINARY_DIR}/testing_venv/bin:$ENV{PATH}")
  endif()
elseif (DEACTIVATE)
  set (ENV{PATH} $ENV{OLD_PATH})
  unset (ENV{OLD_PATH})
  unset (ENV{VIRTUAL_ENV})

else()
  message (WARNING "Incorrect usage of venv activate/deactivate script.\n"
                   "Usage: Specify either ACTIVATE or DEACTIVATE when calling\n"
                   "VALUES: ACTIVATE:${ACTIVATE}; DEACTIVATE:${DEACTIVATE}")
endif()
