# Work around a CUDA 13 nvcc (cudafe++) regression when building downstream
# CUDA extensions (e.g. mmcv) against the installed PyTorch headers.
#
# ATen/core/List_inl.h contains:
#
#   static_cast<typename decltype(impl_->list)::difference_type>(pos)
#
# CUDA 12's nvcc accepted this, but CUDA 13's cudafe++ drops the leading
# 'typename' in its generated host code, so the host compiler then fails with
# "need 'typename' before '...::difference_type' because '...' is a dependent
# scope". The underlying list is std::vector<IValue>, whose difference_type is
# always std::ptrdiff_t, so cast to that directly. This is idempotent: it does
# nothing once the file has already been patched.
#
# Invoked with -DTORCH_INCLUDE_DIR=<torch/include> via cmake -P.

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
