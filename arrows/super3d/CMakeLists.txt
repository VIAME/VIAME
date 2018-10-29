set(header_files
  compute_depth.h
  cost_volume.h
  util.h
  tv_refine_search.h
  warp_image.h
  world_angled_frustum.h
  world_space.h
  )

set(cxx_files
  compute_depth.cxx
  cost_volume.cxx
  util.cxx
  tv_refine_search.cxx
  warp_image.hxx
  warp_image.cxx
  warp_image_instances.cxx
  world_angled_frustum.cxx
  world_space.cxx
  )

kwiver_install_headers(
  SUBDIR     arrows/super3d
  ${header_files}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_super3d_export.h
  NOPATH   SUBDIR     arrows/super3d
  )

kwiver_add_library(kwiver_algo_super3d ${header_files} ${cxx_files})
target_link_libraries(kwiver_algo_super3d kwiver_algo_core kwiver_algo_vxl
  vil vpgl vnl vul vil_algo vgl_algo vbl
  )

algorithms_create_plugin( kwiver_algo_super3d
  register_algorithms.cxx
  )
