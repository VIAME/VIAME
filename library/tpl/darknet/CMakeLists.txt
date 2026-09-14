# darknet, vendored, inference only
#
# VIAME's fork (github.com/VIAME/darknet), reduced to what runs a model.
# `library/object_detectors/darknet_detector` builds an `image_t` and calls
# `Detector::detect`, and that is the whole of what VIAME asks darknet for
# now that training is gone.
#
# What is **not** carried, and why:
#
#   the command line drivers  `darknet.c` and the sixteen subcommands beside
#                             it -- detector, classifier, rnn, go, nightmare,
#                             art, captcha, cifar, coco, compare, dice,
#                             super, swag, tag, voxel, writing, yolo. These
#                             built the `darknet` executable, which is what
#                             `darknet_trainer` used to shell out to. No
#                             training, no executable, no drivers.
#   `demo.c`                  the webcam demo loop
#   `yolo_console_dll.cpp`    a sample program
#
# Two files that look droppable and are not, both found by the linker rather
# than by reading:
#
#   `image_opencv.cpp`  is entirely behind `#ifdef OPENCV` -- except for an
#                       `#else` branch that defines `wait_key_cv`,
#                       `wait_until_press_key_cv`, `resize_window_cv` and
#                       `destroy_all_windows_cv` as no-ops. Training paths in
#                       `network.c` and `network_kernels.cu` call those
#                       **unguarded**, so without this file they are
#                       undefined. Built without `OPENCV`, it is five empty
#                       functions and no OpenCV.
#   `http_stream.cpp`   is not only the MJPEG server: `get_time_point`,
#                       `this_thread_sleep_for` and the `custom_atomic_*`
#                       helpers live there, outside the OpenCV guard, and
#                       `data.c` and the CUDA training loop use them.
#   `gettimeofday.{c,h}`      Windows only
#   `3rdparty/`               getopt and pthreads are Windows only; its stb
#                             is v2.28 and `library/tpl/stb` is v2.30, so
#                             VIAME's copy serves both
#   cfg/, data/, scripts/,    weights, sample images, shell scripts, the
#   docker, the python        python bindings, Docker files, 68 KB of README
#
# 18 MB to 1.8 MB, and the link graph loses OpenCV.
#
# Checked before cutting rather than after: the remaining translation units
# reference nothing in the dropped ones. Training code *inside* the layer
# files stays -- `backward_convolutional_layer` and its kind sit in the same
# translation unit as the forward pass, and splitting them would make this a
# fork of a fork for a few kilobytes.

if( NOT VIAME_ENABLE_DARKNET )
  return()
endif()

file( GLOB darknet_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c" )
file( GLOB darknet_headers "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h" )

# The glob is `*.c`; the three C++ translation units are named, the way
# darknet's own build names them
list( APPEND darknet_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/src/yolo_v2_class.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/image_opencv.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/http_stream.cpp"
  )

if( VIAME_ENABLE_CUDA )
  file( GLOB darknet_cuda_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu" )
endif()

# STATIC, and absorbed into `viame_object_detectors_darknet` rather than
# shipped beside it: one library fewer in the install, and it keeps darknet
# out of every export set -- a shared library linked even PRIVATE still has
# to be exported for consumers to link against the thing that uses it.
add_library( viame_darknet_core STATIC
  ${darknet_sources}
  ${darknet_headers}
  ${darknet_cuda_sources}
  )

target_include_directories( viame_darknet_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

target_include_directories( viame_darknet_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  # `image.c` and `yolo_v2_class.cpp` load images with stb, and VIAME already
  # carries it: darknet's own copy is two patch versions older and there is
  # no reason to have both
  ${CMAKE_SOURCE_DIR}/library/tpl/stb
  )

set_target_properties( viame_darknet_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME darknet
  )

# Darknet reads these to decide whether the GPU paths exist at all. They are
# darknet's own names, not VIAME's, and have to be spelled its way.
if( VIAME_ENABLE_CUDA )
  enable_language( CUDA )
  find_package( CUDAToolkit REQUIRED )

  target_compile_definitions( viame_darknet_core PUBLIC GPU )

  # `cuda_driver` as well as the runtime: `dark_cuda.c` calls
  # `cuCtxGetCurrent`, which is the driver API
  target_link_libraries( viame_darknet_core
    PRIVATE CUDA::cudart CUDA::cublas CUDA::curand CUDA::cuda_driver
    )

  # Darknet's kernels are self-contained, so no separable compilation and no
  # device link step for a consumer to get wrong
  set_target_properties( viame_darknet_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    )

  if( VIAME_ENABLE_CUDNN )
    target_compile_definitions( viame_darknet_core PUBLIC CUDNN )

    # Half precision, which darknet's own build turns **on** by default and
    # forces on for any architecture from 70 up. It is not a tuning knob:
    # with it the convolutions run through cuDNN's tensor-core FP16 path and
    # without it they do not, and the difference shows in the fifth decimal
    # of a confidence. The recorded goldens were made against a fork built
    # this way, and they reproduce exactly only with it.
    set( _darknet_half FALSE )
    foreach( _arch IN LISTS CMAKE_CUDA_ARCHITECTURES )
      string( REGEX REPLACE "[^0-9]" "" _arch_number "${_arch}" )
      if( _arch_number GREATER_EQUAL 70 )
        set( _darknet_half TRUE )
      endif()
    endforeach()

    if( _darknet_half )
      target_compile_definitions( viame_darknet_core PUBLIC CUDNN_HALF )
    endif()

    find_library( CUDNN_LIBRARY cudnn
      HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR}
            /usr/lib/x86_64-linux-gnu )

    if( CUDNN_LIBRARY )
      target_link_libraries( viame_darknet_core PRIVATE ${CUDNN_LIBRARY} )
    else()
      message( FATAL_ERROR "VIAME_ENABLE_CUDNN is on and cudnn was not found" )
    endif()
  endif()
endif()

find_package( Threads REQUIRED )
target_link_libraries( viame_darknet_core PRIVATE Threads::Threads )

find_package( OpenMP )
if( OpenMP_C_FOUND )
  target_link_libraries( viame_darknet_core PRIVATE OpenMP::OpenMP_C )
endif()

# Darknet is C89-era code with a great many unused results and sign
# comparisons. It is vendored, not maintained here, so its warnings are not
# VIAME's to fix and not worth reading past.
target_compile_options( viame_darknet_core PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-w>
  $<$<COMPILE_LANGUAGE:CXX>:-w>
  )

add_library( viame::darknet ALIAS viame_darknet_core )

# Joins the export set: a static library linked into an exported target has
# to be in it too. `export()` writes the build-tree file naming every
# dependency, before a generator expression can hide one, so
# `$<BUILD_INTERFACE:>` on the consumer's side is not enough on its own.
install( TARGETS viame_darknet_core EXPORT ${kwiver_export_name}
         ARCHIVE DESTINATION lib )
set_property( GLOBAL APPEND PROPERTY viame_export_targets viame_darknet_core )
