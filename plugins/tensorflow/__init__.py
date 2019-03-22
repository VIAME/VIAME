
from viame.processes.tf_detector import tf_detector


def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:tf_detector.tf_detector'

    if process_factory.is_process_module_loaded( module_name ):
        return

    process_factory.add_process('tf_detector', 'TF detector',
                                tf_detector.tf_detector )

    process_factory.mark_process_module_as_loaded( module_name )
