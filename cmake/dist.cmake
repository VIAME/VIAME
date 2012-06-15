add_custom_target(dist)
add_custom_command(
  TARGET  dist
  COMMAND "${GIT_EXECUTABLE}"
          archive
          --format=tar
          --prefix=vistk-${vistk_version}/
          v${vistk_version} | bzip2 > vistk-${vistk_version}.tar.bz2
  WORKING_DIRECTORY
          "${vistk_source_dir}"
  COMMENT "Making a tarball of version ${vistk_version}")

add_custom_target(snap)
add_custom_command(
  TARGET  snap
  COMMAND "${CMAKE_COMMAND}"
          -D "vistk_version=${vistk_version}"
          -D "vistk_source_dir=${vistk_source_dir}"
          -D "vistk_binary_dir=${vistk_binary_dir}"
          -D "GIT_EXECUTABLE=${GIT_EXECUTABLE}"
          -P "${vistk_source_dir}/cmake/snap-script.cmake"
  WORKING_DIRECTORY
          "${vistk_source_dir}"
  COMMENT "Making a tarball for the current checkout")
