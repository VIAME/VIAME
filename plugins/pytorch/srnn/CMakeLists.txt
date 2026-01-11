###
# SRNN tracker utilities

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch/srnn
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/track.py
  pytorch/srnn
  track )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/models.py
  pytorch/srnn
  models )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/srnn_matching.py
  pytorch/srnn
  srnn_matching )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_feature_extractor.py
  pytorch/srnn
  siamese_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/iou_tracker.py
  pytorch/srnn
  iou_tracker )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/gt_bbox.py
  pytorch/srnn
  gt_bbox )

###
# SRNN Training modules

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/g_config.py
  pytorch/srnn
  g_config )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/storage.py
  pytorch/srnn
  storage )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/utilities.py
  pytorch/srnn
  utilities )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/ContrastiveLoss.py
  pytorch/srnn
  ContrastiveLoss )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/all_training.py
  pytorch/srnn
  all_training )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_training.py
  pytorch/srnn
  siamese_training )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/rnn_training.py
  pytorch/srnn
  rnn_training )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_dataset.py
  pytorch/srnn
  siamese_dataset )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/rnn_dataset.py
  pytorch/srnn
  rnn_dataset )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/target_rnn_dataset.py
  pytorch/srnn
  target_rnn_dataset )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/train_everything.py
  pytorch/srnn
  train_everything )

###
# SRNN Training executable scripts

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_main_train.py
  pytorch/srnn
  siamese_main_train )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/rnn_main_train.py
  pytorch/srnn
  rnn_main_train )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/target_rnn_main_train.py
  pytorch/srnn
  target_rnn_main_train )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/generate_training_files_kw18.py
  pytorch/srnn
  generate_training_files_kw18 )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/extract_siamese_features.py
  pytorch/srnn
  extract_siamese_features )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/sgi2png.py
  pytorch/srnn
  sgi2png )
