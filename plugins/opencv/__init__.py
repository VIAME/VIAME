# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:opencv.opencv_processes'

    if process_factory.is_process_module_loaded( module_name ):
        return

    from viame.processes.opencv import ocv_multimodal_registration
    from viame.processes.opencv import ocv_fft_filter_based_on_ref

    process_factory.add_process(
        'ocv_multimodal_registration',
        'Register optical and thermal frames',
        ocv_multimodal_registration.register_frames_process
    )

    process_factory.add_process(
        'ocv_fft_filter_based_on_ref',
        'Filter image in the frequency based on some template',
        ocv_fft_filter_based_on_ref.filter_based_on_ref_process
    )

    from . import ocv_stereo_processes
    ocv_stereo_processes.__sprokit_register__()

    process_factory.mark_process_module_as_loaded( module_name )
