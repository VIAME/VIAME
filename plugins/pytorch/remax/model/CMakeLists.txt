add_subdirectory( ops )

kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/attention.py
    arrows/pytorch/remax/model
    attention)
                  
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/backbone.py
    arrows/pytorch/remax/model
    backbone)
                  
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/convnext.py
    arrows/pytorch/remax/model
    convnext)
                  
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/deformable_transformer.py
    arrows/pytorch/remax/model
    deformable_transformer)
                  
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/dino.py
    arrows/pytorch/remax/model
    dino)
                  
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/dn_components.py
    arrows/pytorch/remax/model
    dn_components)
                  
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/matcher.py
    arrows/pytorch/remax/model
    matcher)
                  
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/position_encoding.py
    arrows/pytorch/remax/model
    position_encoding)
                  
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/segmentation.py
    arrows/pytorch/remax/model
    segmentation)
                      
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/swin_transformer.py
    arrows/pytorch/remax/model
    swin_transformer)
                      
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/transformer_deformable.py
    arrows/pytorch/remax/model
    transformer_deformable)
                      
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.py
    arrows/pytorch/remax/model
    utils)