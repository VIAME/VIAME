Configuration Usage
===================

Introduction
------------

The vital ``config_block`` supports general configuration tasks where
a general purpose key/value pair is needed. The configuration block is
used to specify, communicate and set configurable items in many
different situations. The two major users of the configuration support
are algorithms and processes. In addition, there are a few other
places that they are used also.

Configurations are ususally established in an external file which are
read and converted to an internal ``config_block`` object. This is the
typical way to control the behaviour of the software. Configuration
blocks can also be created pragmatically such as when specifying an
expected set of configurable items.

When algorithms are used within processes, the configuration entries
are specified as a block in the pipe file. The process takes the
appropriate config subblock and passes it to the
``set_nested_algo_configuration()`` method to instantiate and
configure the algorithm.


From File to config_block
-------------------------

Using config_block_io to directly convert config file into block. This
can be used by a main program that manages configs and algorithms
directly. The ``read_congfig_file()`` uses a complex set of rules to
locate config files based on host system and application name.


Configuration Features
----------------------

config features. what they do and why you would want to use them
- relativepath

- macro providers and how they can be used to make portable and reusable config files

- config sub-blocks and configuration context


Establishing Expected Config
----------------------------

Typically the expected config is formulated by creating a config block
with all the expected keys, default values, and entry
description. This is done for both algorithms and processes.

Don't be shy with the entry description. This description serves as
the design specification for the entry. The expected format is a short
description followed by a longer detailed description separated by two
new-lines, as shown below.

.. code-block:: c++

  config->set_value( "config_name", <default_value>,
                     "Short description.\n\n"
                     "Longer description which contains all information needed "
                     "to correctly specify this parameter including any range "
                     "limitations etc." );

The long description does not need any new-line characters for
formatting unless specific formatting is desired. The text is wrapped
into a text block by all tools that display it.

This expected configuration serves as documentation for the algorithm
or process configuration items when it is displayed by the
plugin_explorer and other tools. It is also used to validate the
configuration supplied at run time to make sure all expected items are
present.


Usage by Algorithms
'''''''''''''''''''

Algorithms specify their expected set of configurable items in their
``get_configuration()`` method using the config_block ``set_value()`` method,
described above.

The run time configuration is passed to an algorithm through the
``set_configuration()`` method. This method typically extracts the
expected configuration values and saves them locally for the algorithm
to use. When a configuration is read from the file, there is no
guarantee that all expected configuration items are present and
attempting to get a value that is not present generates an exception.

The recommended way to avoid this problem is to use the expected
configuration, as created by the ``get_configuration()`` method to
supply any missing entries. The following code snippet shows how this
is done.

.. code-block:: c++

    // Set this algorithm's properties via a config block
    void
    <algorithm>
    ::set_configuration(vital::config_block_sptr in_config)
    {
      // Starting with our generated vital::config_block to ensure that assumed values are present
      // An alternative is to check for key presence before performing a get_value() call.
      vital::config_block_sptr config = this->get_configuration();

      // Merge in supplied config to cause these values to overwrite the defaults.
      config->merge_config(in_config);

      // Get individual config entry values
      this->write_float_features = config->get_value<bool>("write_float_features",
                                                           this->write_float_features);
    }



Instantiating Algorithms
''''''''''''''''''''''''

Algorithms can be used directly in application code or they can be
wrapped by a sprokit process. In either case the actual implementation
of the abstract algorithm interface is specified through a config block.

Lets first look at the code that will instantiate the configured
algorithm and then look at the contents of the configuration file.

The following code snippet instantiates a ``draw_detected_object_set``
algorithm.

.. code-block:: c++

  // this pointer will be used to reference the algorithm after it is created.
  vital::algo::draw_detected_object_set_sptr m_algo;

  // Get algorithm configuration
  auto algo_config = get_config(); // or an equivalent call

  // Check config so it will give run-time diagnostic of config problems
  if ( ! vital::algo::draw_detected_object_set::check_nested_algo_configuration( "draw_algo", algo_config ) )
  {
    LOG_ERROR( logger, "Configuration check failed." );
  }

  vital::algo::draw_detected_object_set::set_nested_algo_configuration( "draw_algo", algo_config, m_algo );
  if ( ! d->m_algo )
  {
    LOG_ERROR( logger, "Unable to create algorithm." );
  }

After the configuration is extracted, it is passed to the
``check_nested_algo_configuration()`` method to determine if the
configuration has the basic ``type`` entry and the requested type is
available. If the ``type`` entry is missing or the specified
implementation is not available, a detailed log message is generated
with the available implementations.

If the configuration is acceptable, the
``set_nested_algo_configuration()`` call will actually instantiate and
configure the selected algorithm implementation.

The name that is supplied to these calls, "draw_algo" in this case, is
used access the configuration block for this algorithm.

The following configuration file snippet can be used to configure
the above algorithm.::

  block draw_algo
    type = ocv    # select the ocv instance of this algorithm

    block ocv     # configure the 'ocv' instance
      alpha_blend_prob   = true
      default_line_thickness   = 1.25
      draw_text   = false
    endblock # for ocv
  endblock  # for draw_algo

The outer block labeled "draw_algo" specifies the configuration to be
used for the above code snippet. The config entry "type" specifies
which implementation of the algorithm to instantiate. The following
block labeled "ocv" is used to configure the algorithm after it is
instantiated. The block labeled "ocv" is used for algorithm type
"ocv". If the algorithm type was "foo", then the block "foo" would be
used to configure the algorithm.


Usage by Processes
''''''''''''''''''

The configuration for sprokit processes is presented slightly
differently than for algorithms, but underneath, they both use the
same structure.

Configuration items for a process are defined using
``create_config_trait()`` macro as shown below.

.. code-block:: c++

  //                    name,      type,  default,        description
  create_config_trait( threshold, float, "-1", "min threshold for output (float).\n\n"
                       "Detections with confidence values below this threshold are not drawn." );

When the process is constructed all configuration parameters must be
declared using the ``declare_config_using_trait()`` call, as shown below.::

  declare_config_using_trait( threshold );

All configuration items declared in this way are available for display
using the plugin_explorer tool.

Configuration values are extracted from the process configuration in
the ``_configure()`` method of the process as shown below.::

  float local_threshold = config_value_using_trait( threshold );

Processes can instantiate and configure algorithms using the approach
described above.

Configuration for a process comes from a section of the pipe file. The
following section of a pipe file shows configuration for a process
which supplies the threshold configuration item.::

  # ================================
  process draw_boxes :: draw_detected_object_boxes
    threshold = 3.14


Verifying a Configuration
'''''''''''''''''''''''''

When a configuration file (or configuration section of a pipe file) is
read in, there is no checking of the configuration key names. There is
no way of knowing which configuration items are valid or expected and
which ones are not. If a name is misspelled, which sometimes happens,
it will be misspelled in the configuration block. This can lead to
hours of frustration diagnosing a problem.

A configuration can be checked against a baseline using the
config_difference class. This class provides methods to determine the
differences between a reference configuration and one created from an
input file. The difference between these two configurations is
presented in two different ways. It provides a list of keys that are
baseline config and not in the supplied config. These are the config
items that were expected but not supplied. It also provides a list of
keys that are in the supplied config but not in the expected
config. These items are supplied but not expected.

The following code snippet shows how to report the difference between
two config blocks.

.. code-block:: c++

  //                                    ref-config                received-config
  kwiver::vital::config_difference cd( this->get_configuration(), config );
  const auto key_list = cd.extra_keys();
  if ( ! key_list.empty() )
  {
    // This may be considered an error in some cases
    LOG_WARN( logger(), "Additional parameters found in config block that are not required or desired: "
              << kwiver::vital::join( key_list, ", " ) );
  }

  key_list = cd.unspecified_keys();
  if ( ! key_list.empty() )
  {
    LOG_WARN( logger(), "Parameters that were not supplied in the config, using default values: "
              << kwiver::vital::join( key_list, ", " ) );
  }


Not all applications need to check both cases. There may be good
reasons for not specifying all expected configuration items when the
default values are as expected. In some cases, unexpected items that
are supplied by the configuration may be indications of misspelled
entries.


Config Management Techniques
----------------------------

The configuration file reader provides several alternatives for
managing the complexity of a large configuration. The block / endblock
construct can be used to shorted config lines and modularize the
configuration. The include directove can be used to share or reuse
portions of a config.

Starting with the example config section that selects an algorithm and
configures it::

    algorithm_instance_name:type = type_name
    algorithm_instance_name:type_name:algo_param = value
    algorithm_instance_name:type_name:threshold = 234

The block construct can be used to simplify the configuration and
make it easier to navigate.::

  block algorithm_instance_name
    type = type_name
    block  type_name
      algo_param = value
      threshold = 234
    endblock
  endblock

In cases where the configuration block is extensive or used in
multiple applications, that part of the configuration can exist as a
stand-alone file and be included where it is needed.::

    block  algorithm_instance_name
      include type_name.conf
    endblock

where ``type_name.conf`` contains::

    type = type_name
    block   type_name
      algo_param = value
      threshold = 234
    endblock

Environment variables and config macros can be combined to provide a
level of adaptability to config files. Using the environment macro in
an include directive can provide run time agility without requiring
the file to be edited. The following is an example of selecting a
different include file based on mode.::

  include $ENV{MODE}/config.file.conf


Using enums in config entries
-----------------------------

Quite often a configuration parameter can only take a fixed number of
values such as when the user is trying to configure an enum. The enum
support in vital directly supports converting strings to enum values
with the use of the ``enum_converter`` and enum support in the config
block. The enum converter will verify that the supplied string
represents an enum value, and throw an error if it does not. The list
of valid enum strings is provided to assist in documenting config
entries.

The following code snippet shows how to use the enum support to create
a new config entry and convert config entry to an enum value.::

   #include <vital/util/enum_converter.h>

   using kvll = kwiver::vital::kwiver_logger::log_level_t;

   // Declare the enum converter
   //              converter-name   enum-type
   ENUM_CONVERTER( level_converter, kvll,
      { "trace", kvll::LEVEL_TRACE },
      { "debug", kvll::LEVEL_DEBUG },
      { "info",  kvll::LEVEL_INFO },
      { "warn",  kvll::LEVEL_WARN },
      { "error", kvll::LEVEL_ERROR }
    );


    // Create config entry from enum. level_converter supplies the list of
    // valid enum strings.
   conf->set_value( "level", level_converter().to_string( m_log_level ),
                   "Logger level to use when generating log messages. "
                   "Allowable values are: " + level_converter().element_name_string()
    );


   // Convert config entry to an enum value.
   kvll log_level = conf->get_enum_value<level_converter>( "level" );
