*************
KWIVER Traits
*************

In order to create a set of useful processes, port names and data
types must be consistemtly used. KWIVER Traits help establish these
standards.  There are a set of native traits that are supplied with
KWIVER, but it is possible for an application (set of cooperating
processes) to create traits that are specific to an application that
can be overlayed on the native traits.

KWIVER Traits provide a more concise and shareable API for dealing
with ports and config as opposed to the sprokit native API. Even when
using traits, the native API is available and can be used concurrently
with the traits API.

The KWIVER traits are obtained by including the
``kwiver_type_traits.h`` file.

Type Traits
-----------

KWIVER types (also called canonical types) are *logical* not a type
within a programming language. A *double* can represent a distance or
a time interval (or even a distance is a different unit!), but a port
which uses a *double* to a distance would have a type of
*distance_in_meters*, *not* *double*.

These logical types are defined in a common include file using a set
of macros. For example, a typical  data type is the bounding box and
its trait is defined as follows. ::

    create_type_trait( image, "kwiver:image", kwiver::vital::image_container_sptr );

The first parameter is the trait name that will be used when referring
to this type.

The second parameter is the type name string which is used internally
when negotiating connections. This is a free-form string, which can be
used in any desired. KWIVER Traits uses a namespace approach to
prevent type name collisions. In this case the string "kwiver:" is the
namespace specifier. Other applications that are creating type traits
should use an application specific name space.

The third parameter is the language specific specification for the
type of the actual data that will be passed between processes.

Type traits also standardize the actual data type that is used. In
this case of the *image*, the underlying data type is polymorphic, so
it must be passed using a pointer rather than by value.

Port Traits
-----------

Just as with types, a common and predictable set of port names is
essential to quickly creating a pipeline description without having to
lookup every port for each process being used.

With a predictable set of port names, it is easy to remember that any
process that takes in an image, the port name will be **image**.

The logical definition for a port is specified in a port trait as
follows. ::

    create_port_trait( depth_map, image, "Depth map stored in image form." );

The first parameter specifies the name of the port. This name will be
visible as the name of a port that can be connected.

The second parameter is the name of the type trait.

the third parameter is a description of the data expected at the
port. this description should be as long as necessary to sufficiently
describe the data semantics.

As with type traits, the port name indicates the *logical* name of the
port to make sure the data has the expected semantics.

Type traits are declared separately from port traits to allow for
uniquely named ports that use the same type. Images are a good example
of different ports using the same type trait. For example, a process
could process stereo image pairs and would have an input port for the
right and left images.

Using Port Traits
-----------------

Port traits are used when declaring ports and when passing data over
the ports. All trait related functions/macros end with the string
``_using_trait``.
The following are examples or declaring ports. ::

    // Set up for required ports
    sprokit::process::port_flags_t required;
    sprokit::process::port_flags_t optional;

    required.insert( flag_required ); // indicate port is required.

    // input ports
    declare_input_port_using_trait( timestamp, required );
    declare_input_port_using_trait( image, required );
    declare_input_port_using_trait( homography_src_to_ref, optional );

    // output ports
    declare_output_port_using_trait( image, optional );

The first parameter is the port trait name.

The second parameter is the variable name of the port flag (of type
``sprokit::process::port_flags_t``). This variable must be defined and
configured prior to declaring the port.

The next data element can be retreived from a port using the
''grab_from_port_using_trait()'' as follows. ::

  kwiver::vital::timestamp ts;
  ts = grab_from_port_using_trait( timestamp );

This call gets the next element from the port as defined by the
*timestamp* port trait and will ensure that the data item from the
port is the expected type (a timestamp) or an exception is thrown.

Similarly, data can be pushed to an output port using ``push_to_port_using_trait()``
as follows ::

    push_to_port_using_trait( image, out_image );

This call puts the value from ``out_image`` to the ``image`` port. An
exception will be thrown if the value is not of the expected type.

There are more functions available for accessing and manipulating data
on ports. Refer to the documentation on trait_utils for more details.

Config Traits
-------------

Config traits serve to provide a single point where all required
attributes of a configuration item can be declared abd simplifies
declaring and accessing configuration items. Config traits are defined
as follows. ::

    create_config_trait( frame_time, double, "0.03333333", "Inter frame time in seconds. "
    create_config_trait( algorithm, std::string, "", "Name of algorithm config sub-block.\n\n"
                         "Typical usage is:\n"
                         "algorithm = <algo-name>\n"
                         "block <algo-name>\n"
                         "  type = foo\n"
                         "  block foo\n"
                         "    param = val\n"
                         "  endblock  # foo\n"
                         "endblock  # <algo-name>\n" );

The first parameter is the name of the trait and is also the name of
the config entry. This name will be used when declaring and accessing
the config item.

The second parameter is the concrete data type of the config
entry. The config entry value will be converted to thsi type if
possible. Conversions are avaialble for all native types. Refer to the
config block documentation if you need to convert a composite datat
type.

The third parameter is the default value for the config entry if it is
found in the config block. This value must be specified as a string.

The fourth parameter is the description of this config entry. Please
be descriptive since this description is shown by the plugin_explorer
as part of the process documentation.


Using Config Traits
-------------------

Config traits are usually defined in the process that needs a specific
config element, although there may be some very common traits already
defined in the ``kwiver_type_traits.h`` file.

Creating a config trait using ``create_config_trait()`` only creates a
static data structure so this needs to be done in the implementation
file, within any namespace blocks, but before any code, just as with
any other static data items.

Configuration items need to be declared when a process is
created. This is usually done in the ``make_config()`` method, but can
be done directly in the constructor. Configuration items are declared
as follows. ::

    declare_config_using_trait( frame_time );

The declare function only takes the name of the pre-defined trait.

The value from the process config is retrieved as follows. ::

    m_frame_time = config_value_using_trait( frame_time );

This call retrieves the configured value as specified in the pipe file
or returns the default value if it has not been specified. The  value
has been converted to the type specified when the trait was created.

There are more functions for working with config traits that can be
found in the documentation for the ``kwiver_type_traits.h`` file.
