###
# The namespace names VIAME had before phase 11
#
# Header only, and nothing in VIAME includes it: it is installed for code
# outside VIAME that still writes `kwiver::vital` or `sprokit`. Open decision
# 8 keeps it for one release.
##

viame_install_headers( kwiver.h SUBDIR viame/compat )

###
# The python package name VIAME had before phase 11
#
# The counterpart to `kwiver.h` above: `kwiver.vital.types` and the rest
# still import, and resolve to the module at its new name.
#
# Installed by hand rather than through `viame_add_python_module`, because
# that helper installs into the `viame` package and this one has to be top
# level -- being called `kwiver` is the whole point of it.
##
if( VIAME_ENABLE_PYTHON )
  set( _viame_kwiver_shim "${CMAKE_CURRENT_SOURCE_DIR}/kwiver" )

  foreach( _viame_shim_file IN ITEMS __init__.py aliases.py )
    configure_file(
      "${_viame_kwiver_shim}/${_viame_shim_file}"
      "${kwiver_python_output_path}/${python_sitename}/kwiver/${_viame_shim_file}"
      COPYONLY
      )
    install(
      FILES       "${_viame_kwiver_shim}/${_viame_shim_file}"
      DESTINATION "${python_site_packages}/kwiver"
      )
  endforeach()
endif()
