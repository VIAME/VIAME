Example C++ external module creation.

This example compiles a C++ shared library (.so or .dll) which contains
a loadable VIAME C++ object detector. The difference between this and other
examples is that it can link against an existing VIAME install and is
built outside of the VIAME build chain as opposed to inside of it. In order
to build it, you need to set the VIAME_DIR cmake variable to the location
of a VIAME install, but in this example VIAME need not be built from source,
only this plugin.
