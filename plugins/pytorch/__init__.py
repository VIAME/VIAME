# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from viame.processes.pytorch import convert_to_onnx_process
from viame.processes.pytorch import foundation_stereo_process

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:pytorch.pytorch_processes'

    if process_factory.is_process_module_loaded( module_name ):
        return

    process_factory.add_process(
        'convert_to_onnx',
        'Convert a yolo/cfrcnn model to onnx',
        convert_to_onnx_process.OnnxConverter
    )

    process_factory.add_process(
        'foundation_stereo',
        'Stereo disparity estimation using NVIDIA Foundation-Stereo model',
        foundation_stereo_process.FoundationStereoProcess
    )

    process_factory.mark_process_module_as_loaded( module_name )
