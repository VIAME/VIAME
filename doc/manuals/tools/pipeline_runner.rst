===============
Pipeline Runner
===============

The pipeline runner applet is used to execute a preconfigured pipeline application and is invoked as follows:

kwiver runner [options] pipe-file
---------------------------------

**Options are:**

  ``-h, --help``

    Display applet usage.

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

  ``-S, --scheduler <type>``

    Specifies the scheduler type to use.

The ``pipe-file`` is the name of the pipeline defintion file.
See <xxx> for a description of the pipeline syntax.
