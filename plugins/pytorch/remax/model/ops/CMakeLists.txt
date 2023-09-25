kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/ms_deform_attn.py
    arrows/pytorch/remax/model/ops/modules
    ms_deform_attn)
                  
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/functions/ms_deform_attn_func.py
    arrows/pytorch/remax/model/ops/functions
    ms_deform_attn_func)
                  