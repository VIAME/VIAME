# custom_patch_pytorch_nccl.cmake
#
# Disable PyTorch's NCCL symmetric-memory support.
#
# The NCCL bundled with PyTorch (2.29.7) exposes a symmetric-memory device API
# that does not compile under CUDA 12.8: nccl_device.h instantiates OpSum<half>
# during its host pass, where half has no operator+, so nvcc fails in NCCL's
# reduce_copy__types.h. That breaks NCCLSymmetricMemory.cu, nccl_extension.cu and
# ops/nccl_reduce_scatter_offset.cu -> torch_cuda -> the entire build.
#
# Both the #include and the four capability macros must go. The macros gate every
# use of the device API (all #ifdef NCCL_HAS_SYMMEM_*), but the #include sits in a
# version check rather than a capability check, so dropping the macros alone still
# pulls in the uncompilable header. With both removed the affected translation
# units compile to empty bodies. ncclWindow_t and friends come from nccl.h and are
# unaffected, as are standard NCCL collectives and DDP.
#
# Revisit when NCCL or CUDA is bumped.
#
# Required variables:
#   PYTORCH_SOURCE_DIR - Root of the PyTorch checkout

cmake_minimum_required( VERSION 3.16 )

if( NOT PYTORCH_SOURCE_DIR )
  message( FATAL_ERROR "custom_patch_pytorch_nccl.cmake requires PYTORCH_SOURCE_DIR" )
endif()

set( HEADER
  "${PYTORCH_SOURCE_DIR}/torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp" )

if( NOT EXISTS "${HEADER}" )
  message( STATUS "PyTorch NCCL symmem header not found, skipping patch" )
  return()
endif()

set( TAG "// VIAME_DISABLE_NCCL_SYMMEM (does not compile under CUDA 12.8):" )

file( READ "${HEADER}" CONTENTS )

if( CONTENTS MATCHES "VIAME_DISABLE_NCCL_SYMMEM" )
  message( STATUS "PyTorch NCCL symmem already disabled (no patching needed)" )
  return()
endif()

# Split into lines by hand so that anchors apply per line, protecting any
# semicolons in the source from being read as list separators on the way
string( REPLACE ";" "<SEMICOLON>" CONTENTS "${CONTENTS}" )
string( REPLACE "\n" ";" LINES "${CONTENTS}" )

set( MACROS
  "NCCL_HAS_SYMMEM_SUPPORT|NCCL_HAS_SYMMEM_DEVICE_SUPPORT|NCCL_HAS_ONE_SIDED_API|NCCL_DEVICE_HAS_REDUCE_COPY" )

set( PATCHED_LINES )
set( PATCHED_COUNT 0 )

foreach( LINE IN LISTS LINES )
  string( REGEX REPLACE "\r$" "" STRIPPED "${LINE}" )

  if( STRIPPED MATCHES "^#define (${MACROS})$" OR
      STRIPPED STREQUAL "#include <nccl_device.h>" )
    list( APPEND PATCHED_LINES "${TAG} ${LINE}" )
    math( EXPR PATCHED_COUNT "${PATCHED_COUNT} + 1" )
  else()
    list( APPEND PATCHED_LINES "${LINE}" )
  endif()
endforeach()

if( NOT PATCHED_COUNT EQUAL 5 )
  message( FATAL_ERROR
    "Expected to disable 4 NCCL symmem macros + 1 include, got ${PATCHED_COUNT}\n"
    "  ${HEADER} has likely changed upstream; the patch needs review" )
endif()

string( REPLACE ";" "\n" PATCHED "${PATCHED_LINES}" )
string( REPLACE "<SEMICOLON>" ";" PATCHED "${PATCHED}" )

file( WRITE "${HEADER}" "${PATCHED}" )

message( STATUS "PyTorch NCCL symmem patching complete" )
