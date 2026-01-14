# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:pytorch.pytorch_processes'

    if process_factory.is_process_module_loaded( module_name ):
        return

    # Note: srnn_tracker, deepsort_tracker, botsort_tracker, siammask_tracker,
    # mdnet_tracker are now vital algorithms registered via
    # __vital_algorithm_register__ and used with the track_objects process

    try:
        from viame.pytorch import torchvision_augment_process
        torchvision_augment_process.__sprokit_register__()
    except ImportError:
        pass

    try:
        from viame.pytorch import convert_to_onnx_process
        process_factory.add_process(
            'convert_to_onnx',
            'Convert a yolo/cfrcnn model to onnx',
            convert_to_onnx_process.OnnxConverter
        )
    except ImportError:
        pass

    process_factory.mark_process_module_as_loaded( module_name )


def __vital_algorithm_register__():
    """Register vital algorithm implementations."""
    try:
        from viame.pytorch import srnn_tracker
        srnn_tracker.__vital_algorithm_register__()
    except ImportError:
        pass

    try:
        from viame.pytorch import deepsort_tracker
        deepsort_tracker.__vital_algorithm_register__()
    except ImportError:
        pass

    try:
        from viame.pytorch import botsort_tracker
        botsort_tracker.__vital_algorithm_register__()
    except ImportError:
        pass

    try:
        from viame.pytorch import siammask_tracker
        siammask_tracker.__vital_algorithm_register__()
    except ImportError:
        pass

    try:
        from viame.pytorch import sam3_tracker
        sam3_tracker.__vital_algorithm_register__()
    except ImportError:
        pass

    try:
        from viame.pytorch import mdnet_tracker
        mdnet_tracker.__vital_algorithm_register__()
    except ImportError:
        pass
