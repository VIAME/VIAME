#
# INSERT COPYRIGHT STATEMENT OR DELETE THIS
#

"""What this package provides, without importing any of it.

VIAME reads these lists when the package is named in `VIAME_PYTHON_PLUGINS`
and registers a stand-in for each entry, which imports its module the first
time a pipeline asks for it.
"""

# ( interface, name, description, "module:Class" )
__vital_algorithm_declarations__ = [
    ( "image_filter", "example_filter",
      "Example externally created python image filter",
      "example_external_plugin.example_filter:ExampleFilter" ),
]

# ( name, description, "module:Class" )
__sprokit_process_declarations__ = [
    ( "example_filter_process",
      "Example externally created python process",
      "example_external_plugin.example_filter_process:ExampleFilterProcess" ),
]
