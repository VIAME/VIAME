# Root package
viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch/siammask
  __init__ )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/siammask_trainer.py
  pytorch/siammask
  siammask_trainer )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/gen_json.py
  pytorch/siammask
  gen_json )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/par_crop.py
  pytorch/siammask
  par_crop )

# Core module
viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/core/__init__.py
  pytorch/siammask/core
  __init__ )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/core/config.py
  pytorch/siammask/core
  config )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/core/xcorr.py
  pytorch/siammask/core
  xcorr )

# Models module
viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/__init__.py
  pytorch/siammask/models
  __init__ )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/init_weight.py
  pytorch/siammask/models
  init_weight )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/loss.py
  pytorch/siammask/models
  loss )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/model_builder.py
  pytorch/siammask/models
  model_builder )

# Models backbone
viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/backbone/__init__.py
  pytorch/siammask/models/backbone
  __init__ )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/backbone/alexnet.py
  pytorch/siammask/models/backbone
  alexnet )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/backbone/mobile_v2.py
  pytorch/siammask/models/backbone
  mobile_v2 )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/backbone/resnet_atrous.py
  pytorch/siammask/models/backbone
  resnet_atrous )

# Models head
viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/head/__init__.py
  pytorch/siammask/models/head
  __init__ )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/head/rpn.py
  pytorch/siammask/models/head
  rpn )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/head/mask.py
  pytorch/siammask/models/head
  mask )

# Models neck
viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/neck/__init__.py
  pytorch/siammask/models/neck
  __init__ )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/neck/neck.py
  pytorch/siammask/models/neck
  neck )

# Tracker module
viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/__init__.py
  pytorch/siammask/tracker
  __init__ )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/base_tracker.py
  pytorch/siammask/tracker
  base_tracker )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/siamrpn_tracker.py
  pytorch/siammask/tracker
  siamrpn_tracker )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/siammask_tracker.py
  pytorch/siammask/tracker
  siammask_tracker )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/siamrpnlt_tracker.py
  pytorch/siammask/tracker
  siamrpnlt_tracker )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/tracker_builder.py
  pytorch/siammask/tracker
  tracker_builder )

# Utils module
viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/__init__.py
  pytorch/siammask/utils
  __init__ )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/bbox.py
  pytorch/siammask/utils
  bbox )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/model_load.py
  pytorch/siammask/utils
  model_load )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/log_helper.py
  pytorch/siammask/utils
  log_helper )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/average_meter.py
  pytorch/siammask/utils
  average_meter )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/misc.py
  pytorch/siammask/utils
  misc )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/anchor.py
  pytorch/siammask/utils
  anchor )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/lr_scheduler.py
  pytorch/siammask/utils
  lr_scheduler )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/distributed.py
  pytorch/siammask/utils
  distributed )

# Datasets module
viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/datasets/__init__.py
  pytorch/siammask/datasets
  __init__ )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/datasets/augmentation.py
  pytorch/siammask/datasets
  augmentation )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/datasets/anchor_target.py
  pytorch/siammask/datasets
  anchor_target )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/datasets/dataset.py
  pytorch/siammask/datasets
  dataset )
