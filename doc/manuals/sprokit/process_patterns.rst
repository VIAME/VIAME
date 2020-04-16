Process Coding Patterns
=======================

Generally the best parctices for writing processes is captured by the
template process, but there are some features that are not universally
needed in all processes. The following code snippets are the best
practice for some frequent operations in a process.

Defining a Loadable Module Using PLUGIN_INFO()
----------------------------------------------

All plugins need to be defined by using the PLUGIN_INFO() macro in the
class declaration in the ehader file. An example for an algorithm
follows but the concepts are the same for all plugins.::

   PLUGIN_INFO( "render-mesh",
                 "Render a depth or height map from a mesh.\n\n"
                 "This tool reads in a mesh file and a camera and renders "
                 "various images such as depth map or height map.");

The first argument specifies the name of the plugin and the second is
the documentation describing the plugin. The documentation starts with
a short one line description, not to exceed 65 characters, followed by
a blank line. The short description should be a sentence, in that it
starts with a capital letter and ends with a period. The blank line
separates the short description from the functional
specification. This specification should be as long as needed to
provide enough information to allow somebody to use it without having
to resort to perusing the source code. Do not add new-line characters
unless a hard return is needed.  This text will be wrapped when
displayed.

Verifying Config Parameters
---------------------------

The following code snippet can be used in the
``process::_configure()`` method to report any unexpected
configuration parameters. Unexpected parameters are usually a
misspelling of a valid parameter. Tracking these down can take quite a
while. If your process has a well defined set of configuration
parameters, this code can be used to spot misspellings. Note that
processes that wrap algorithms (arrows) can not usually do this since
the configuration of the arrow is not known to the process.::

    #include <vital/config/config_difference.h>

    void
    your_process
    ::_configure()
    {
      // check for extra config keys
      auto cd = this->config_diff();
      cd.warn_extra_keys( logger() );

      // regular config processing
    }


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
