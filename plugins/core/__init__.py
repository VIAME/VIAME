# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from viame.processes.core import utility_processes

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:core.core_processes'

    if process_factory.is_process_module_loaded( module_name ):
        return

    process_factory.add_process(
        'blank_out_frames',
        'Blank out frames with no object detections on them',
        utility_processes.blank_out_frames
    )

    process_factory.add_process(
        'percentile_norm_npy_16_to_8bit',
        'A specialized percentile normalization method',
        utility_processes.percentile_norm_npy_16_to_8bit
    )

    try:
        from viame.processes.core import bytetrack_tracker
        bytetrack_tracker.__sprokit_register__()
    except ImportError:
        pass

    try:
        from viame.processes.core import ocsort_tracker
        ocsort_tracker.__sprokit_register__()
    except ImportError:
        pass

    try:
        from viame.processes.core import simple_homog_tracker
        simple_homog_tracker.__sprokit_register__()
    except ImportError:
        pass

    try:
        from viame.processes.core import multicam_homog_tracker
        multicam_homog_tracker.__sprokit_register__()
    except ImportError:
        pass

    try:
        from viame.processes.core import stabilize_many_images
        stabilize_many_images.__sprokit_register__()
    except ImportError:
        pass

    try:
        from viame.processes.core import multicam_homog_det_suppressor
        multicam_homog_det_suppressor.__sprokit_register__()
    except ImportError:
        pass

    process_factory.mark_process_module_as_loaded( module_name )
