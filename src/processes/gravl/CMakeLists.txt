project(vistk_processes_gravl)

find_package(gravl REQUIRED)

# Check if C++11 support is required by gravl, if not using MSVC. (On MSVC,
# there are no options for it, so nothing we can do without knowing the
# respective compiler versions; we will be okay as long as we are using the
# same or never version as was used to build gravl; otherwise gravl's
# build-time check catch any incompatibility).
if (NOT MSVC)
  if (GRAVL_REQUIRE_CPP11)
    if (NOT has_cxx11_flags)
      message(SEND_ERROR
        "Found gravl built with C++11 support, but your compiler has no option "
        "to enable C++11 support. This configuration is not supported.")
    elseif (NOT VISTK_ENABLE_CXX11)
      message(SEND_ERROR
        "Found gravl built with C++11 support, but VISTK_ENABLE_CXX11 is FALSE. "
        "This configuration is not supported.")
    endif ()
  endif ()
endif ()

include_directories(SYSTEM "${gravl_INCLUDE_DIR}")

set(gravl_srcs
  read_video_gravl_process.cxx
  registration.cxx)

set(gravl_private_headers
  gravl-config.h
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
