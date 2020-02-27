===========
pipe-to-dot
===========

The pipe-to-dot applet generates a DOT file from the pipeline topology.
This graphical output is useful for visualizing and documenting a specific pipeline.

kwiver pipe-to-dot  [options]
-----------------------------

**Options are:**

  ``-h, --help``

    Display applet usage

**input options:**

  ``-p, --pipe-file arg``

    Selects a pipeline file to render as a DOT file.

  ``-C, --cluster <arg>``

    Selects a cluster file to render as a DOT file.

  ``-T, --cluster-type <arg>``

    Selects a cluster type to render as a DOT file. This is the name of the cluster
    in the cluster definition file.

  Note: Only one input option is allowed.

**output options:**

  ``-n, --name <arg>``

    Specifies the name of the output graph.

  ``-o, --output <file>``

    Name of output file or '-' for stdout. (default: -)


  ``-P, --link-prefix <arg>``

    Prefix for links when formatting for sphinx.


**pipe options:**

  ``-c, --config <file>``

    Specifies a file containing supplemental configuration entries.
    Can occur multiple times on the command line.

    The supplemental configuration files can contain additional
    configuration entires for processes or global config
    blocks. Configuration items for processes start with the process
    name. For example The ``input_path`` config element for the process
    ``reader`` would be specified as follows:

    ``reader:input_path = input/file/path``

    These configuration values are applied to the pipeline configuration
    after the pipeline file has been processes and can overwrite values
    previously defined in the pipe file.

  ``-s, --settings VAR=VALUE``

    Specifies additional configuration settings to be applied to the pipeline configuration.

    The settings option can be used to specify a single configuration
    value. This command line option can appear multiple times on the
    command line. Using the same example above, that configuration item can be
    specified as follows:

    ``--settings reader:input_path=input/file/path``

    Note that there can be no embedded spaces in the argument.

  ``-I, --include <path>``

    Specifies a directory to be added to the configuration include path.
    This option can occur multiple times on the command line

  ``--setup``

  Setup pipeline before rendering


Example
-------

Usage example showing DOT rendering tools
