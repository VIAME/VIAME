Example C++ external module creation.

This example compiles a C++ shared library (.so or .dll) which contains
a loadable VIAME C++ object detector, registered as
``external_example_detector`` -- a name of its own, since a name VIAME already
registers is an error. The difference between this and other
examples is that it can link against an existing VIAME install and is
built outside of the VIAME build chain as opposed to inside of it. In order
to build it, you need to set the VIAME_DIR cmake variable to the location
of a VIAME install, but in this example VIAME need not be built from source,
only this plugin.

Loading it
----------

VIAME registers everything it was built with by calling it directly -- there
is no plugin directory and nothing is searched for. A plugin built outside
the tree is named instead, in ``VIAME_PLUGIN_PATH``::

    export VIAME_PLUGIN_PATH=$PWD/lib/modules/example_plugin.so
    viame registry-dump --json | grep external_example_detector

The variable holds a list of **files**, separated by ``:`` on Unix and ``;``
on Windows, not a list of directories. Each is opened and asked for
``viame_register_plugin``; a file that cannot be opened, or that does not
export that function, is logged and skipped, and the rest of the list is
still loaded.
