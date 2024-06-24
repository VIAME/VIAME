
from viame.processes.external_example import example_filter

def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:viame.example_filter'

    if process_factory.is_process_module_loaded( module_name ):
      return

    process_factory.add_process( 'example_filter',
      'Example external filter', example_filter )

    process_factory.mark_process_module_as_loaded( module_name )
