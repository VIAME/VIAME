
# Create python init for siammask package
kwiver_create_python_init( pytorch/siammask )
kwiver_create_python_init( pytorch/siammask/core )
kwiver_create_python_init( pytorch/siammask/models )
kwiver_create_python_init( pytorch/siammask/models/backbone )
kwiver_create_python_init( pytorch/siammask/models/head )
kwiver_create_python_init( pytorch/siammask/models/neck )
kwiver_create_python_init( pytorch/siammask/tracker )
kwiver_create_python_init( pytorch/siammask/utils )
kwiver_create_python_init( pytorch/siammask/datasets )

# Root package
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch/siammask
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/siammask_trainer.py
  pytorch/siammask
  siammask_trainer )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/gen_json.py
  pytorch/siammask
  gen_json )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/par_crop.py
  pytorch/siammask
  par_crop )

# Core module
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/core/__init__.py
  pytorch/siammask/core
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/core/config.py
  pytorch/siammask/core
  config )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/core/xcorr.py
  pytorch/siammask/core
  xcorr )

# Models module
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/__init__.py
  pytorch/siammask/models
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/init_weight.py
  pytorch/siammask/models
  init_weight )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/loss.py
  pytorch/siammask/models
  loss )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/model_builder.py
  pytorch/siammask/models
  model_builder )

# Models backbone
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/backbone/__init__.py
  pytorch/siammask/models/backbone
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/backbone/alexnet.py
  pytorch/siammask/models/backbone
  alexnet )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/backbone/mobile_v2.py
  pytorch/siammask/models/backbone
  mobile_v2 )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/backbone/resnet_atrous.py
  pytorch/siammask/models/backbone
  resnet_atrous )

# Models head
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/head/__init__.py
  pytorch/siammask/models/head
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/head/rpn.py
  pytorch/siammask/models/head
  rpn )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/head/mask.py
  pytorch/siammask/models/head
  mask )

# Models neck
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/neck/__init__.py
  pytorch/siammask/models/neck
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/models/neck/neck.py
  pytorch/siammask/models/neck
  neck )

# Tracker module
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/__init__.py
  pytorch/siammask/tracker
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/base_tracker.py
  pytorch/siammask/tracker
  base_tracker )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/siamrpn_tracker.py
  pytorch/siammask/tracker
  siamrpn_tracker )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/siammask_tracker.py
  pytorch/siammask/tracker
  siammask_tracker )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/siamrpnlt_tracker.py
  pytorch/siammask/tracker
  siamrpnlt_tracker )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/tracker_builder.py
  pytorch/siammask/tracker
  tracker_builder )

# Utils module
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/__init__.py
  pytorch/siammask/utils
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/bbox.py
  pytorch/siammask/utils
  bbox )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/model_load.py
  pytorch/siammask/utils
  model_load )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/log_helper.py
  pytorch/siammask/utils
  log_helper )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/average_meter.py
  pytorch/siammask/utils
  average_meter )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/misc.py
  pytorch/siammask/utils
  misc )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/anchor.py
  pytorch/siammask/utils
  anchor )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/lr_scheduler.py
  pytorch/siammask/utils
  lr_scheduler )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/distributed.py
  pytorch/siammask/utils
  distributed )

# Datasets module
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/datasets/__init__.py
  pytorch/siammask/datasets
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/datasets/augmentation.py
  pytorch/siammask/datasets
  augmentation )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/datasets/anchor_target.py
  pytorch/siammask/datasets
  anchor_target )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/datasets/dataset.py
  pytorch/siammask/datasets
  dataset )
