###
# Release packaging: `cpack -G TGZ`
#
# What this replaces on Linux is `prepare_linux_desktop_install`,
# `create_install_tarball` and `restore_linux_desktop_install` in
# `build_common_functions.sh`: move seven directories out of the install
# prefix, tar what is left, move them back.
#
# It archives the finished install prefix, as the tarball did, rather than
# having CPack install into a staging tree of its own. Most of an install is
# not made by install rules: pip installs the python dependencies into the
# prefix during the build (`viame_python_deps.cmake`, `viame_python_forks.cmake`),
# the CPython build installs there, and so does DIVE -- about 83,000 of an
# install's 85,000 files. A package built from install rules alone would have
# no torch and no DIVE.
#
# What a binary release leaves out is decided in `viame-package-prune.cmake.in`,
# run on CPack's copy before it is archived: the directories the tarball step
# moved aside, and `include/` and `lib/cmake`, which are for building a plugin
# against the install -- `VIAME_PACKAGE_DEVEL` keeps those two (open decision
# 10 kept the out-of-tree plugin hook). `lite-install-size.md` section 3 row 4.
#
#   cmake --build <build> --target install
#   cd <build> && cpack -G TGZ -D CPACK_PACKAGE_FILE_NAME=VIAME-<version>-<platform>
#
# The archive's top directory is `viame/`, as the tarball's was.
##

option( VIAME_PACKAGE_DEVEL
  "Keep include/ and lib/cmake in cpack packages, for building plugins against them" OFF )
mark_as_advanced( VIAME_PACKAGE_DEVEL )

set( CPACK_PACKAGE_NAME                "VIAME" )
set( CPACK_PACKAGE_VENDOR              "Kitware" )
set( CPACK_PACKAGE_VERSION             "${VIAME_VERSION}" )
set( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Video and Image Analytics for Marine Environments" )
set( CPACK_RESOURCE_FILE_LICENSE       "${VIAME_SOURCE_DIR}/LICENSE.txt" )
set( CPACK_PACKAGE_FILE_NAME           "VIAME-${VIAME_VERSION}-${CMAKE_SYSTEM_NAME}" )

set( CPACK_GENERATOR                   "TGZ" )
set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY  OFF )

# The finished prefix, copied under `viame/`; no install rules are run.
set( CPACK_INSTALL_CMAKE_PROJECTS      "" )
set( CPACK_INSTALLED_DIRECTORIES       "${VIAME_BUILD_INSTALL_PREFIX};viame" )

set( _viame_package_script "${CMAKE_BINARY_DIR}/viame-package-prune.cmake" )
configure_file(
  "${VIAME_CMAKE_DIR}/viame-package-prune.cmake.in"
  "${_viame_package_script}"
  @ONLY )
set( CPACK_PRE_BUILD_SCRIPTS "${_viame_package_script}" )

include( CPack )
