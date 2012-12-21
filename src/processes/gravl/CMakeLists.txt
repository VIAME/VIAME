project(vistk_processes_gravl)

include_directories(${gravl_INCLUDE_DIR})

set(gravl_srcs
  read_video_gravl_process.cxx
  registration.cxx)

set(gravl_private_headers
  read_video_gravl_process.h
  registration.h)

vistk_private_header_group(${gravl_private_headers})
add_vistk_plugin(processes_gravl
  MAKE_VISTK_PROCESSES_GRAVL_LIB
  ${gravl_srcs}
  ${gravl_private_headers})
target_link_libraries(processes_gravl
  vistk_pipeline
  image_helper
  gravl_raf
  gravl_core)
