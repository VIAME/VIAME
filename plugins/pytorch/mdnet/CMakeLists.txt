
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/bbreg.py
  pytorch/mdnet
  bbreg )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/data_prov.py
  pytorch/mdnet
  data_prov )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/img_cropper.py
  pytorch/mdnet
  img_cropper )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch/mdnet
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/model.py
  pytorch/mdnet
  model )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/options.py
  pytorch/mdnet
  options )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/prepro_data.py
  pytorch/mdnet
  prepro_data )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/pretrain_options.py
  pytorch/mdnet
  pretrain_options )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/sample_generator.py
  pytorch/mdnet
  sample_generator )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.py
  pytorch/mdnet
  utils )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker.py
  pytorch/mdnet
  tracker )
