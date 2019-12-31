
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/bbreg.py
  viame/arrows/pytorch/mdnet
  bbreg )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/data_prov.py
  viame/arrows/pytorch/mdnet
  data_prov )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/img_cropper.py
  viame/arrows/pytorch/mdnet
  img_cropper )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  viame/arrows/pytorch/mdnet
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/model.py
  viame/arrows/pytorch/mdnet
  model )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/options.py
  viame/arrows/pytorch/mdnet
  options )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/prepro_data.py
  viame/arrows/pytorch/mdnet
  prepro_data )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/prepro_data_imagenet.py
  viame/arrows/pytorch/mdnet
  prepro_data_imagenet )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/pretrain_options.py
  viame/arrows/pytorch/mdnet
  pretrain_options )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/sample_generator.py
  viame/arrows/pytorch/mdnet
  sample_generator )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.py
  viame/arrows/pytorch/mdnet
  utils )

