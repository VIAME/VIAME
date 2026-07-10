message( "Running pytorch auxiliary install" )

# CUDA 13 nvcc (cudafe++) fix for the installed torch ATen/core/List_inl.h
# header. It contains:
#
#   static_cast<typename decltype(impl_->list)::difference_type>(pos)
#
# CUDA 12's nvcc accepted this, but CUDA 13's cudafe++ drops the leading
# 'typename' in its generated host code, so the host compiler fails with
# "need 'typename' before '...::difference_type'", breaking downstream CUDA
# extension builds (e.g. mmcv). The underlying list is std::vector<IValue>,
# whose difference_type is always std::ptrdiff_t, so cast to that directly.
# Only runs when invoked with -DTORCH_INCLUDE_DIR=<torch/include> (set by the
# pytorch source build step for CUDA 13+); idempotent.
if( DEFINED TORCH_INCLUDE_DIR )
  set( _list_inl "${TORCH_INCLUDE_DIR}/ATen/core/List_inl.h" )
  if( EXISTS "${_list_inl}" )
    file( READ "${_list_inl}" _contents )
    string( REPLACE
      "static_cast<typename decltype(impl_->list)::difference_type>(pos)"
      "static_cast<std::ptrdiff_t>(pos)"
      _patched "${_contents}" )
    if( NOT "${_patched}" STREQUAL "${_contents}" )
      file( WRITE "${_list_inl}" "${_patched}" )
      message( STATUS "Patched ${_list_inl} for CUDA 13 nvcc compatibility" )
    else()
      message( STATUS "${_list_inl} already CUDA 13 compatible, no patch needed" )
    endif()
  else()
    message( STATUS "${_list_inl} not found, skipping CUDA 13 nvcc patch" )
  endif()
endif()

if( WIN32 )

  # Patches directly to pytorch
  set( TORCH_DIR ${VIAME_PYTHON_BASE}/site-packages/torch )

  if( NOT EXISTS ${TORCH_DIR} )
    set( TORCH_DIR ${VIAME_PYTHON_BASE}/dist-packages/torch )
  endif()

  if( VIAME_PYTORCH_VERSION VERSION_EQUAL "1.7.1" )
    set( PATCH ${VIAME_PATCH_DIR}/pytorch/mmcv-win32-1.7.1/include )
  elseif( VIAME_PYTORCH_VERSION VERSION_EQUAL "1.4.0" )
    set( PATCH ${VIAME_PATCH_DIR}/pytorch/mmcv-win32-1.4.0/include )
  endif()

  if( EXISTS ${TORCH_DIR} )
    file( COPY ${PATCH}
          DESTINATION ${TORCH_DIR} )
  endif()

endif()

message( "Done" )
