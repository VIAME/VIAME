###
# Loadable algorithms

set(kwiver_python_subdir "python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")
set(kwiver_python_output_path "${KWIVER_BINARY_DIR}/lib/${kwiver_python_subdir}")

kwiver_create_python_plugin_init(kwiver/arrows/pytorch)

# TODO: change KWIVER_ENABLE_PYTHON to KWIVER_ENABLE_PYTORCH
if( KWIVER_ENABLE_PYTHON )

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/SRNN_tracking.py
    kwiver/processes
	SRNN_tracking)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/seg_detection.py
    kwiver/processes
	seg_detection)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/pytorch_siamese_f_extractor.py
    kwiver/arrows/pytorch
    pytorch_siamese_f_extractor)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/grid.py
    kwiver/arrows/pytorch
    grid)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/track.py
    kwiver/arrows/pytorch
    track)
  
  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/models.py
    kwiver/arrows/pytorch
    models)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/SRNN_matching.py
    kwiver/arrows/pytorch
	SRNN_matching)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/MOT_bbox.py
    kwiver/arrows/pytorch
	MOT_bbox)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/seg_utils.py
    kwiver/arrows/pytorch
	seg_utils)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/fcn_segmentation.py
    kwiver/arrows/pytorch
	fcn_segmentation)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/fcn_models.py
    kwiver/arrows/pytorch
	fcn_models)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/iou_tracking.py
    kwiver/arrows/pytorch
	iou_tracking)
endif()
