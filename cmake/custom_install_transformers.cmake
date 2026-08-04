message( "Running transformers auxiliary install" )

# transformers 5.x has module-level `torch.distributed` imports that are gated
# only on the torch *version* (is_torch_greater_or_equal) or is_torch_available(),
# but NOT on torch.distributed.is_available(). VIAME's torch is built without
# distributed support (no torch._C._distributed_c10d), so these imports hard-fail
# at import time and surface as misleading transformers lazy-loader errors such as
# "Could not import module 'AutoBackbone'", which breaks rf-detr / HuggingFace
# backbone loading. Each guard below is extended to also require
# torch.distributed.is_available(). Both patches are idempotent (the matched text
# no longer exists once patched) and skip cleanly when the file or the target
# transformers version is not present.

set( _tf_dir "${VIAME_PYTHON_BASE}/site-packages/transformers" )
if( NOT EXISTS "${_tf_dir}" )
  set( _tf_dir "${VIAME_PYTHON_BASE}/dist-packages/transformers" )
endif()

if( NOT EXISTS "${_tf_dir}" )
  message( STATUS "transformers not installed, skipping distributed-guard patch" )
  return()
endif()

# --- Patch 1: transformers/distributed/fsdp.py -------------------------------
set( _fsdp "${_tf_dir}/distributed/fsdp.py" )
if( EXISTS "${_fsdp}" )
  file( READ "${_fsdp}" _contents )
  string( REPLACE
    [=[if is_torch_available() and is_torch_greater_or_equal("2.6"):]=]
    [=[if is_torch_available() and is_torch_greater_or_equal("2.6") and torch.distributed.is_available():]=]
    _patched "${_contents}" )
  if( NOT "${_patched}" STREQUAL "${_contents}" )
    file( WRITE "${_fsdp}" "${_patched}" )
    message( STATUS "Patched ${_fsdp} distributed guard" )
  else()
    message( STATUS "${_fsdp} already patched or not applicable, skipping" )
  endif()
else()
  message( STATUS "${_fsdp} not found, skipping" )
endif()

# --- Patch 2: transformers/distributed/sharding_utils.py ---------------------
set( _shard "${_tf_dir}/distributed/sharding_utils.py" )
if( EXISTS "${_shard}" )
  # Anchor on the preceding `if is_torch_available():` so only the runtime import
  # block is matched, not the identical two lines under `if TYPE_CHECKING:`.
  file( READ "${_shard}" _contents )
  string( REPLACE
[=[if is_torch_available():
    import torch
    from torch.distributed.tensor import DTensor]=]
[=[if is_torch_available():
    import torch

if is_torch_available() and torch.distributed.is_available():
    from torch.distributed.tensor import DTensor]=]
    _patched "${_contents}" )
  if( NOT "${_patched}" STREQUAL "${_contents}" )
    file( WRITE "${_shard}" "${_patched}" )
    message( STATUS "Patched ${_shard} distributed guard" )
  else()
    message( STATUS "${_shard} already patched or not applicable, skipping" )
  endif()
else()
  message( STATUS "${_shard} not found, skipping" )
endif()

message( "Done" )
