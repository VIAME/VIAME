add_subdirectory( ops )

viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    pytorch/remax/model
    __init__)

viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/attention.py
    pytorch/remax/model
    attention)
                  
viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/backbone.py
    pytorch/remax/model
    backbone)
                  
viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/convnext.py
    pytorch/remax/model
    convnext)
                  
viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/deformable_transformer.py
    pytorch/remax/model
    deformable_transformer)
                  
viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/dino.py
    pytorch/remax/model
    dino)
                  
viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/dn_components.py
    pytorch/remax/model
    dn_components)
                  
viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/matcher.py
    pytorch/remax/model
    matcher)
                  
viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/position_encoding.py
    pytorch/remax/model
    position_encoding)
                  
viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/segmentation.py
    pytorch/remax/model
    segmentation)
                      
viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/swin_transformer.py
    pytorch/remax/model
    swin_transformer)
                      
viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/transformer_deformable.py
    pytorch/remax/model
    transformer_deformable)
                      
viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.py
    pytorch/remax/model
    utils)
