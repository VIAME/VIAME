function (sprokit_configure_pkgconfig module)
  if (UNIX)
    set(pkgconfig_file "${sprokit_binary_dir}/lib/pkgconfig/${module}.pc")

    sprokit_configure_file(sprokit-${module}.pc
      "${CMAKE_CURRENT_SOURCE_DIR}/${module}.pc.in"
      "${pkgconfig_file}"
      sprokit_version
      CMAKE_INSTALL_PREFIX
      LIB_SUFFIX
      ${ARGN})

    sprokit_install(
      FILES       "${pkgconfig_file}"
      DESTINATION "lib${LIB_SUFFIX}/pkgconfig"
      COMPONENT   development)
  endif ()
endfunction ()
