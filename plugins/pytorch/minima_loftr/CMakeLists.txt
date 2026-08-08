###
# Vendored MINIMA-LoFTR matcher (inference only) — used by the interactive
# alignment (auto-align) backend for cross-modality (e.g. EO<->IR) image
# registration. See __init__.py for provenance and local modifications.

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch/minima_loftr
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/loftr.py
  pytorch/minima_loftr
  loftr )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/backbone/__init__.py
  pytorch/minima_loftr/backbone
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/backbone/resnet_fpn.py
  pytorch/minima_loftr/backbone
  resnet_fpn )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/loftr_module/__init__.py
  pytorch/minima_loftr/loftr_module
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/loftr_module/fine_preprocess.py
  pytorch/minima_loftr/loftr_module
  fine_preprocess )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/loftr_module/linear_attention.py
  pytorch/minima_loftr/loftr_module
  linear_attention )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/loftr_module/transformer.py
  pytorch/minima_loftr/loftr_module
  transformer )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/utils/__init__.py
  pytorch/minima_loftr/utils
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/utils/coarse_matching.py
  pytorch/minima_loftr/utils
  coarse_matching )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/utils/fine_matching.py
  pytorch/minima_loftr/utils
  fine_matching )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/utils/position_encoding.py
  pytorch/minima_loftr/utils
  position_encoding )
