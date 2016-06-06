add_custom_target(dist)
add_custom_command(
  TARGET  dist
  COMMAND "${GIT_EXECUTABLE}"
          archive
          --format=tar
          --prefix=sprokit-${KWIVER_VERSION}/
          v${KWIVER_VERSION} | bzip2 > sprokit-${KWIVER_VERSION}.tar.bz2
  WORKING_DIRECTORY
          "${sprokit_source_dir}"
  COMMENT "Making a tarball of version ${KWIVER_VERSION}")

add_custom_target(snap)
add_custom_command(
  TARGET  snap
  COMMAND "${CMAKE_COMMAND}"
          -D "KWIVER_VERSION=${KWIVER_VERSION}"
          -D "sprokit_source_dir=${sprokit_source_dir}"
          -D "sprokit_binary_dir=${sprokit_binary_dir}"
          -D "GIT_EXECUTABLE=${GIT_EXECUTABLE}"
          -P "${sprokit_source_dir}/extra/dist/snap-script.cmake"
  WORKING_DIRECTORY
          "${sprokit_source_dir}"
  COMMENT "Making a tarball for the current checkout")
