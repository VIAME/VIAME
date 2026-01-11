# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:pytorch.pytorch_processes'

    if process_factory.is_process_module_loaded( module_name ):
        return

    from viame.processes.pytorch import srnn_tracker
    srnn_tracker.__sprokit_register__()

    try:
        from viame.processes.pytorch import siammask_tracker
        siammask_tracker.__sprokit_register__()
    except ImportError:
        pass

    try:
        from viame.processes.pytorch import mdnet_tracker_process
        mdnet_tracker_process.__sprokit_register__()
    except ImportError:
        pass

    try:
        from viame.processes.pytorch import desc_augmentation_process
        desc_augmentation_process.__sprokit_register__()
    except ImportError:
        pass

    try:
        from viame.processes.pytorch import convert_to_onnx_process
        process_factory.add_process(
            'convert_to_onnx',
            'Convert a yolo/cfrcnn model to onnx',
            convert_to_onnx_process.OnnxConverter
        )
    except ImportError:
        pass

    process_factory.mark_process_module_as_loaded( module_name )
