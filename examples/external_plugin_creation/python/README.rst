Example external python script which can be called in a VIAME pipeline.

After making a python script there are two ways to run it, it can either
be placed in the [VIAME_INSTALL]/lib/python3.6/*-packages/viame/processes
directory in a way similar to the other python modules in there, or
alternatively, you can append the location of the script to your PYTHONPATH
environment variable.

Note: the included CMakeLists.txt file isn't actually required to run the
python filter in VIAME, it simply copies the python library into the correct
folder to run it so that VIAME's plugin manager picks up the file as an
example.

There are two ways to implement plugins, either as a derived class of a
base algorithm class (e.g. image_filter) which has a simpler API, or
as a custom pipelined process. The former is usually neater and easier,
though the later allows more customization of inputs and outputs to
a particular algorithm in case it's necessary.
