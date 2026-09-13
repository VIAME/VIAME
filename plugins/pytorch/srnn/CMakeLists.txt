###
# SRNN tracker utilities

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch/srnn
  __init__ )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/track.py
  pytorch/srnn
  track )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/models.py
  pytorch/srnn
  models )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/srnn_matching.py
  pytorch/srnn
  srnn_matching )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_feature_extractor.py
  pytorch/srnn
  siamese_feature_extractor )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/iou_tracker.py
  pytorch/srnn
  iou_tracker )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/gt_bbox.py
  pytorch/srnn
  gt_bbox )

###
# SRNN Training modules

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/g_config.py
  pytorch/srnn
  g_config )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/storage.py
  pytorch/srnn
  storage )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/utilities.py
  pytorch/srnn
  utilities )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/ContrastiveLoss.py
  pytorch/srnn
  ContrastiveLoss )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/all_training.py
  pytorch/srnn
  all_training )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_training.py
  pytorch/srnn
  siamese_training )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/rnn_training.py
  pytorch/srnn
  rnn_training )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_dataset.py
  pytorch/srnn
  siamese_dataset )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/rnn_dataset.py
  pytorch/srnn
  rnn_dataset )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/target_rnn_dataset.py
  pytorch/srnn
  target_rnn_dataset )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/train_everything.py
  pytorch/srnn
  train_everything )

###
# SRNN Training executable scripts

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_main_train.py
  pytorch/srnn
  siamese_main_train )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/rnn_main_train.py
  pytorch/srnn
  rnn_main_train )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/target_rnn_main_train.py
  pytorch/srnn
  target_rnn_main_train )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/generate_training_files.py
  pytorch/srnn
  generate_training_files )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/extract_siamese_features.py
  pytorch/srnn
  extract_siamese_features )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/extract_features_shard.py
  pytorch/srnn
  extract_features_shard )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/sgi2png.py
  pytorch/srnn
  sgi2png )
