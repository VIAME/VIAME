# general notes on running a pipeline

## Building boost for Python support

Boost can not be built with debug mode because it activates asserts
that prevent modules from loading.

## General module loading

### Modules/Processes

Modules are loaded from a set of directories that are configured into the loader at build time.
The default directory is "lib/sprokit" subdirectory of the install tree. Additionally, at build time,
The build directory can be added to the loading path by selecting the cmake variable KWIVER_USE_BUILD_TREE

Set environment variable SPROKIT_MODULE_PATH to list directories of loadable modules

### Clusters

Cluster files define groups of processes that behave like a single process. Cluster definition files
end with the .cluster extension.

Clusters are loaded from "share/sprokit/pipelines/clusters" subdirectory of the install tree.
Set environment variable SPROKIT_CLUSTER_PATH to list directories of cluster configurations.

The build directory can be added to the cluster path if the cmake variable SPROKIT_USE_BUILD_CLUSTER_PATH is enabled.

## PYTHON

Set environment PYTHONPATH to contain location of loadable modules.

Typically install/lib/python2.7/dist-packages
or        install/lib/python2.7/site-packages

The python path is initialized in the generated setup script file.

The environment variable SPROKIT_NO_PYTHON_MODULES can be set to suppress loading python modules.
This is only for sprokit module loader.

Additional modules can be loaded by adding them to the SPROKIT_PYTHON_MODULES environment variable

    export SPROKIT_PYTHON_MODULES=kwiver.processes:foo.processes:test.schedulers

### python processes

Once processes are loaded, they need to be registered. Each modile needs to have a `__sprokit_register__`
method to register one or more python processes.

```
def __sprokit_register__():
    from sprokit.pipeline import process_registry

    module_name = 'python:kwiver'  # name of the module

    reg = process_registry.ProcessRegistry.self()

    if reg.is_process_module_loaded(module_name):
        return

    reg.add_process(<process type name>, <description>, <class>)

    reg.mark_module_as_loaded(module_name)
```

Where:
    < process type name > - is a string with the process type name. This
    name is used in the pipeline config file as the process type.

    < description > - is a string defining the process. Consider this documentation.

    < class > - python class name. Class must be derived from sprokit.pipeline.process.PythonProcess


# Process-o-pedia

The process listing will need some structure. A flexible arbitrary
hierarchy would be nice. Could be taken from the loading
structure. That is, list processes grouped by .so that registered
them. Sorted alphabetically in the groups.

# Schedulers

Schedulers can be configured by creating a special configuration block
with the name "_scheduler". This configuration block is then presented
to the actual scheduler being used to run the pipeline. The valid
configuration for schedulers currently depends on the scheduler
selected.

```
config _scheduler
  :num_thread 23

```


# Debugging support

Make separate schedulers with extensive instrumentation and those with
none. Maybe a scheduler decorator. Does not look like this will get
what we need.  Maybe a strategy pattern where there are calls at
critical points that can be ignored or generate instrumentation.

The goal is to not have two identical schedulers, one with
instrumentaiton, the other without.

Instrumentation for RightTrack
Extra log messages for tracing.

Instrumenting the process looks like a better approach than
instrumenting the scheduler.

# Python interfacing

Needs examples of python processes.

Needs writeup of theory of conversions to/from python and sprokit
ports. Use real struct converters as examples.

Needs writeup on theory of vital C wrappers and how to convert from
sprokit to python.

Improvement: Make any converters a class, not just two unbound
functions. The class provides a more functor like environment where
there can be state and better debugging. Specifically each converter
can make its target type visible.

A major problem with this conversion stuff is (in addition to the
boost::python magic) that it is opaque.
