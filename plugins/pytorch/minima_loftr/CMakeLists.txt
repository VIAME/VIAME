###
# Vendored MINIMA-LoFTR matcher (inference only) — used by the interactive
# alignment (auto-align) backend for cross-modality (e.g. EO<->IR) image
# registration. See __init__.py for provenance and local modifications.

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch/minima_loftr
  __init__ )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/loftr.py
  pytorch/minima_loftr
  loftr )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/backbone/__init__.py
  pytorch/minima_loftr/backbone
  __init__ )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/backbone/resnet_fpn.py
  pytorch/minima_loftr/backbone
  resnet_fpn )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/loftr_module/__init__.py
  pytorch/minima_loftr/loftr_module
  __init__ )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/loftr_module/fine_preprocess.py
  pytorch/minima_loftr/loftr_module
  fine_preprocess )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/loftr_module/linear_attention.py
  pytorch/minima_loftr/loftr_module
  linear_attention )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/loftr_module/transformer.py
  pytorch/minima_loftr/loftr_module
  transformer )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/utils/__init__.py
  pytorch/minima_loftr/utils
  __init__ )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/utils/coarse_matching.py
  pytorch/minima_loftr/utils
  coarse_matching )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/utils/fine_matching.py
  pytorch/minima_loftr/utils
  fine_matching )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/utils/position_encoding.py
  pytorch/minima_loftr/utils
  position_encoding )
