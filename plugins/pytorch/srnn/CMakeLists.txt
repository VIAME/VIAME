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

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/__init__.py
  pytorch/srnn/training
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/g_config.py
  pytorch/srnn/training
  g_config )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/storage.py
  pytorch/srnn/training
  storage )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/utilities.py
  pytorch/srnn/training
  utilities )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/ContrastiveLoss.py
  pytorch/srnn/training
  ContrastiveLoss )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/all_training.py
  pytorch/srnn/training
  all_training )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/siamese_training.py
  pytorch/srnn/training
  siamese_training )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/rnn_training.py
  pytorch/srnn/training
  rnn_training )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/siamese_dataset.py
  pytorch/srnn/training
  siamese_dataset )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/rnn_dataset.py
  pytorch/srnn/training
  rnn_dataset )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/target_rnn_dataset.py
  pytorch/srnn/training
  target_rnn_dataset )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/train_everything.py
  pytorch/srnn/training
  train_everything )

###
# SRNN Training executable scripts

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/bin/__init__.py
  pytorch/srnn/training/bin
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/bin/siamese_main_train.py
  pytorch/srnn/training/bin
  siamese_main_train )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/bin/rnn_main_train.py
  pytorch/srnn/training/bin
  rnn_main_train )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/bin/target_rnn_main_train.py
  pytorch/srnn/training/bin
  target_rnn_main_train )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/bin/generate_training_files_kw18.py
  pytorch/srnn/training/bin
  generate_training_files_kw18 )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/bin/extract_siamese_features.py
  pytorch/srnn/training/bin
  extract_siamese_features )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/training/bin/sgi2png.py
  pytorch/srnn/training/bin
  sgi2png )
