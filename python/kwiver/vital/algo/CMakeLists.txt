set( THIS_MODULE vital/algo )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py "${THIS_MODULE}" __init__ )

################################################################################

# simple macro to add class.h, class.cxx and class_trampoline.txx to a list for
# each class in class_list
macro(kwiver_expand_pybind_class_list class_list  sources_var)
  foreach(class IN ITEMS ${class_list})
    list(APPEND ${sources_var} "${CMAKE_CURRENT_BINARY_DIR}/${class}.h")
    list(APPEND ${sources_var} "${CMAKE_CURRENT_BINARY_DIR}/${class}.cxx")
    list(APPEND ${sources_var} "${CMAKE_CURRENT_BINARY_DIR}/${class}_trampoline.txx")
  endforeach()
endmacro()

# The interface headers these bindings are generated from live in VIAME's
# library/ since phase 5. A class whose header did not come across is skipped
# here as well: the generated list below is kwiver's, and the reachability
# list is what decided which of them VIAME reaches.
macro(kwiver_generate_dependencies class_list sources_var)
  foreach(class IN ITEMS ${class_list})
    set( _header
      "${VIAME_LITE_LIBRARY_DIR}/algorithm_framework/algo/${class}.h" )
    if( EXISTS "${_header}" )
      list(APPEND ${sources_var} "${_header}")
    endif()
  endforeach()
endmacro()


# Kwiver's list; the ones whose interface did not come across are dropped
# just below, so that this stays comparable with upstream.
set( classes
  activity_detector
  analyze_tracks
  associate_detections_to_tracks
  bundle_adjust
  buffered_metadata_filter
  close_loops
  compute_association_matrix
  compute_depth
  compute_ref_homography
  compute_stereo_depth_map
  compute_track_descriptors
  convert_image
  data_serializer
  detect_features
  detect_motion
  detected_object_filter
  detected_object_set_input
  detected_object_set_output
  draw_detected_object_set
  draw_tracks
  dynamic_configuration
  estimate_canonical_transform
  estimate_essential_matrix
  estimate_fundamental_matrix
  estimate_homography
  estimate_pnp
  estimate_similarity_transform
  extract_descriptors
  feature_descriptor_io
  filter_features
  filter_tracks
  image_filter
  image_io
  image_object_detector
  initialize_cameras_landmarks
  initialize_object_tracks
  integrate_depth_maps
  interpolate_track
  keyframe_selection
  match_descriptor_sets
  match_features
  merge_detections
  merge_images
  metadata_filter
  metadata_map_io
  optimize_cameras
  perform_text_query
  read_object_track_set
  read_track_descriptor_set
  refine_detections
  refine_tracks
  resection_camera
  segment_via_points
  split_image
  track_features
  track_objects
  train_detector
  train_tracker
  transform_2d_io
  triangulate_landmarks
  uuid_factory
  uv_unwrap_mesh
  video_input
  video_output
  write_object_track_set
  write_track_descriptor_set
)

# Drop the ones whose interface did not come across with phase 5's import:
# nothing in VIAME reaches them, so there is nothing to generate a binding
# from. Reported, so the difference from upstream is visible.
set( _viame_lite_classes )
set( _viame_lite_skipped )

foreach( class IN LISTS classes )
  if( EXISTS "${VIAME_LITE_LIBRARY_DIR}/algorithm_framework/algo/${class}.h" )
    list( APPEND _viame_lite_classes ${class} )
  else()
    list( APPEND _viame_lite_skipped ${class} )
  endif()
endforeach()

if( _viame_lite_skipped )
  message( STATUS "vital.algo bindings: not imported, skipped: ${_viame_lite_skipped}" )
endif()

set( classes ${_viame_lite_classes} )

# castxml parses the headers with its own bundled clang -- 13, in the castxml
# 0.4.5 wheel -- while emulating an MSVC compiler, so the standard library it
# reads has to be one that clang can parse. The VS 2026 headers are not: they
# use C++23 static call operators unconditionally and static_assert on any
# clang older than 19, so emulating the compiler that is doing the build fails
# with a few dozen errors out of <memory>, <vector> and <tuple>. Point this at
# an older toolset's cl.exe to parse against its headers instead. Only
# declarations are read from it; the wrappers are still compiled by
# CMAKE_CXX_COMPILER.
set( KWIVER_PYBIND11_CASTXML_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH
  "MSVC compiler for castxml to emulate when generating pybind11 wrappers" )
mark_as_advanced( KWIVER_PYBIND11_CASTXML_COMPILER )

# The emulated compiler only supplies predefined macros. castxml takes the
# system headers from the INCLUDE environment instead, which the build sets to
# CMAKE_CXX_COMPILER's toolset, so emulating an older compiler against newer
# headers still reads the headers that cannot be parsed. Put the emulated
# toolset's own include directories at the front of the parse path, where they
# shadow whatever INCLUDE carries.
set(KWIVER_PYBIND11_PARSER_INCLUDE_DIRS "")
if (MSVC AND NOT KWIVER_PYBIND11_CASTXML_COMPILER STREQUAL CMAKE_CXX_COMPILER)
  get_filename_component(_kwiver_castxml_toolset
    "${KWIVER_PYBIND11_CASTXML_COMPILER}/../../../.." ABSOLUTE)
  foreach (_dir "${_kwiver_castxml_toolset}/include"
                "${_kwiver_castxml_toolset}/ATLMFC/include")
    if (EXISTS "${_dir}")
      string(APPEND KWIVER_PYBIND11_PARSER_INCLUDE_DIRS "${_dir};")
    endif ()
  endforeach ()
endif ()

# generate configuration file for the wrapper generator
set(KWIVER_PYBIND11_ALGO "${classes}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.ini.in  ${CMAKE_CURRENT_BINARY_DIR}/config.ini.in @ONLY)
# due to https://github.com/CastXML/pygccxml/issues/180 we need to run castxml externally
if (MSVC)
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/config.ini.in
"run_external_castxml=1
compiler_path=${KWIVER_PYBIND11_CASTXML_COMPILER}
")
else ()
# Elsewhere, drive castxml ourselves as well, but with no compiler to emulate
# at all. Left to itself pygccxml always asks castxml to emulate a host
# compiler, choosing clang++ if it can find one and falling back to c++, and
# emulating GCC 13 against glibc 2.39 is a combination castxml cannot handle:
# it stubs _Float32/_Float64/_Float128 out as opaque structs that glibc's own
# iseqsig and strtof32 overloads no longer match, giving 20 errors out of
# <cmath>, <cwchar> and <cstdlib> and no xml file. castxml's own bundled clang
# parses the same headers cleanly, and it still finds the system libstdc++ by
# detecting the GCC installation, so emulating nothing is both simpler and
# more portable than requiring a clang on PATH for the generator's benefit.
#
# castxml's clang still has to find a standard library, though, and it only
# looks where a GCC installation would normally be. Hand it the include
# directories the build compiler itself uses so a toolchain kept somewhere
# else -- a Red Hat gcc-toolset under /opt/rh, say -- is found as well.
# Only the standard library directories, though. The rest of
# CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES is either somewhere clang already
# looks or, worse, GCC's own compiler-internal include directory: its SIMD
# intrinsic headers are written against __builtin_ia32_* builtins that clang
# does not have under those names, so putting that ahead of clang's own
# <xmmintrin.h> is 20 errors out of Eigen before any of our headers are read.
set(_kwiver_castxml_include_dirs ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
list(FILTER _kwiver_castxml_include_dirs INCLUDE REGEX "/c\\+\\+/")
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/config.ini.in
"run_external_castxml=1
compiler_include_dirs=${_kwiver_castxml_include_dirs}
")
unset(_kwiver_castxml_include_dirs)
endif ()
file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/config.ini
  INPUT  ${CMAKE_CURRENT_BINARY_DIR}/config.ini.in)


# Interfaces whose trampoline is written by hand rather than generated: the
# generator passes every argument to python by value, which is wrong for the
# in/out parameters these three have. See trampolines/README.md.
set(handwritten_trampolines)
foreach( class IN LISTS classes )
  set( _tramp
    "${CMAKE_CURRENT_SOURCE_DIR}/trampolines/${class}_trampoline.txx" )
  if( EXISTS "${_tramp}" )
    list( APPEND handwritten_trampolines "${_tramp}" )
  endif()
endforeach()
unset( _tramp )

if( handwritten_trampolines )
  message( STATUS "vital.algo bindings: hand-written trampolines: "
                  "${handwritten_trampolines}" )
endif()

# files to generate
set(algo_sources)
kwiver_expand_pybind_class_list("${classes}" algo_sources)
# headers the command depends on
set(vital_algo_dependencies)
kwiver_generate_dependencies("${classes}" vital_algo_dependencies)

# The generator finds castxml by searching PATH, so the install bin directory
# has to be prepended to it. Windows separates PATH entries with ';', not ':',
# and using the wrong one glues the install directory onto the front of the
# first system entry, leaving neither path valid and castxml unfindable.
if( WIN32 )
  set( _kwiver_pybind11_path "${CMAKE_INSTALL_PREFIX}/bin;$ENV{PATH}" )
else()
  set( _kwiver_pybind11_path "${CMAKE_INSTALL_PREFIX}/bin:$ENV{PATH}" )
endif()

# we generate all wrappings at once instead of per file since the startup cost
# of the C++ parser is high
add_custom_command(OUTPUT ${algo_sources}
  COMMAND  ${CMAKE_COMMAND} -E env
          # castxml's entry point imports its own python module, so the
          # site-packages directory is what belongs on the path, which is
          # what `kwiver_python_install_path` is.
          "PYTHONPATH=${kwiver_python_install_path}"
          "PATH=${_kwiver_pybind11_path}"
          ${Python_EXECUTABLE}
          "${CMAKE_CURRENT_SOURCE_DIR}/cpp_to_pybind11.py"
          -c "${CMAKE_CURRENT_BINARY_DIR}/config.ini"
          --handwritten-trampoline-directory
          "${CMAKE_CURRENT_SOURCE_DIR}/trampolines"
  DEPENDS ${vital_algo_dependencies}
          ${handwritten_trampolines}
          "${CMAKE_CURRENT_BINARY_DIR}/config.ini"
          "${CMAKE_CURRENT_SOURCE_DIR}/cpp_to_pybind11.py"
  COMMENT "Generating pybind11 wrappings of algorithm classes"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM
)

# generate algorithm_module.cxx
#------------------------------------------------------------------
set(_kwiver_python_algo_all_modules_includes)
foreach (_kwiver_python_module IN LISTS classes)
  string(APPEND _kwiver_python_algo_all_modules_includes "#include <python/kwiver/vital/algo/${_kwiver_python_module}.h>\n")
endforeach ()

set(_kwiver_python_algo_all_modules_load_algos)
foreach (_kwiver_python_module IN LISTS classes)
  string(APPEND _kwiver_python_algo_all_modules_load_algos "   ${_kwiver_python_module}(m);\n")
endforeach ()

# Hand-written adjustments layered over the generated bindings. These run
# after every generated loader so they can look classes up on the module and
# re-def what the generator cannot express (an in/out parameter, say).
string(APPEND _kwiver_python_algo_all_modules_includes
  "#include <python/kwiver/vital/algo/extract_descriptors_extras.h>\n")
string(APPEND _kwiver_python_algo_all_modules_load_algos
  "   extract_descriptors_extras(m);\n")
string(APPEND _kwiver_python_algo_all_modules_includes
  "#include <python/kwiver/vital/algo/estimator_extras.h>\n")
string(APPEND _kwiver_python_algo_all_modules_load_algos
  "   estimator_extras(m);\n")

set(_kwiver_python_algo_all_modules_content)
string(CONCAT _kwiver_python_algo_all_modules_content
"#include <pybind11/pybind11.h>\n"
"#include <python/kwiver/vital/algo/algorithm.h>\n"
"#include <python/kwiver/vital/algo/algorithm_capabilities.h>\n"
"${_kwiver_python_algo_all_modules_includes}"
"namespace kwiver::vital::python {\n"
"PYBIND11_MODULE(algos,m)\n"
"{\n"
"   algorithm(m);\n"
"   algorithm_capabilities(m);\n"
"${_kwiver_python_algo_all_modules_load_algos}"
"}\n"
"} // namespace \n"
)

file(GENERATE
     OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/algorithm_module.cxx"
     CONTENT "${_kwiver_python_algo_all_modules_content}")

#------------------------------------------------------------------


set(module_sources
  algorithm.h
  algorithm.cxx
  algorithm_capabilities.cxx
  algorithm_trampoline.txx
  out_parameter.txx
  extract_descriptors_extras.cxx
  estimator_extras.cxx
  ${handwritten_trampolines}
  "${CMAKE_CURRENT_BINARY_DIR}/algorithm_module.cxx"
  ${algo_sources}
)

# Python Module Library: _algos
kwiver_add_python_library(
  algos
  "${THIS_MODULE}"
  SOURCES ${module_sources}
  PRIVATE pybind11::pybind11
          vital
          vital_config
          vital_algo
)
