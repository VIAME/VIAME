kwiver_create_python_init( pytorch/remax/model/ops )

kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/__init__.py
    pytorch/remax/model/ops/modules
    __init__)

kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/ms_deform_attn.py
    pytorch/remax/model/ops/modules
    ms_deform_attn)

kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/functions/__init__.py
    pytorch/remax/model/ops/functions
    __init__)

kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/functions/ms_deform_attn_func.py
    pytorch/remax/model/ops/functions
    ms_deform_attn_func)
                  