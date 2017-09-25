/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS [yas] elisp error!AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _KWIVER_TRAIT_UTILS_H_
#define _KWIVER_TRAIT_UTILS_H_

#include <vital/config/config_block_types.h>
#include <sprokit/pipeline/process.h>

//
// These macros are designed to assist in establishing a system wide
// set of consistent types, config keys, and ports.
// Create trait once, use many times.
//
/**
 * \brief Create a configuration item trait.
 *
 * This macro defines a trait object for a single configuration item.
 * The \c NAME parameter defines the name of the trait that is
 * different from the configuration entry key. This form is useful if
 * the configuration key name contains characters that are not allowed
 * in a c++ symbol.

 * Once a configuration trait is created, it can be
 * used to declare a configuration item for a process.
 *
 * The following statement defines a trait called "mission_id".
 \code
 create_config_trait( mission_id, "mission:id", std::string, "", "Mission ID to store in archive" );
 \endcode
 *
 * This trait can be used to define a configuration item for a process as follows:
 \code
  declare_config_using_trait( mission_id );
 \endcode
 *
 * The configuration value can be retrieved from the process config as follows:
 \code
  std::string m_mission_id;
  m_mission_id = config_value_using_trait( mission_id );
 \endcode
 *
 * \param NAME The name of this configuration key.
 * \param KEY Configuration item key name. This must be quoted.
 * \param TYPE Data type for this configuration item.
 * \param DEF Default value for this configuration item.
 * \param DESCR Description of configuration item.
 */
#define create_named_config_trait(NAME, KEY, TYPE, DEF, DESCR)          \
  namespace  { struct NAME ## _config_trait {                           \
  static const kwiver::vital::config_block_key_t      key;              \
  static const kwiver::vital::config_block_value_t    def;              \
  static const kwiver::vital::config_block_description_t description;   \
  typedef TYPE type;                                                    \
};                                                                      \
kwiver::vital::config_block_key_t const NAME ## _config_trait::key = kwiver::vital::config_block_key_t( KEY ); \
kwiver::vital::config_block_value_t const NAME ## _config_trait::def = kwiver::vital::config_block_value_t( DEF ); \
kwiver::vital::config_block_description_t const  NAME ## _config_trait::description = kwiver::vital::config_block_description_t( DESCR ); }

//@{
/**
 * \brief Create a configuration item trait.
 *
 * This macro defines a trait object for a single configuration item.
 * The \c KEY parameter defines the name of the trait and also the
 * configuration key. Once a configuration trait is created, it can be
 * used to declare a configuration item for a process.
 *
 * The following statement defines a trait called "mission_id".
 \code
 create_config_trait( mission_id, std::string, "", "Mission ID to store in archive" );
 \endcode
 *
 * This trait can be used to define a configuration item for a process as follows:
 \code
  declare_config_using_trait( mission_id );
 \endcode
 *
 * The configuration value can be retrieved from the process config as follows:
 \code
  std::string m_mission_id;
  m_mission_id = config_value_using_trait( mission_id );
 \endcode
 *
 * \param KEY Configuration item key name. Also the trait name.
 * \param TYPE Data type for this configuration item.
 * \param DEF Default value for this configuration item. This must be a suitable initializer for a string.
 * \param DESCR Description of configuration item. This must be a suitable initializer for a string.
 */
#define create_config_trait(KEY, TYPE, DEF, DESCR) create_named_config_trait( KEY, # KEY, TYPE, DEF, DESCR )


#define declare_config_using_trait(KEY)                         \
declare_configuration_key( KEY ## _config_trait::key,           \
                           KEY ## _config_trait::def,           \
                           KEY ## _config_trait::description)

#define declare_tunable_config_using_trait(KEY)                         \
declare_configuration_key( KEY ## _config_trait::key,                   \
                           KEY ## _config_trait::def,                   \
                           KEY ## _config_trait::description, true)     \
//@}

// Get value from process config using trait
#define config_value_using_trait(KEY) config_value< KEY ## _config_trait::type >( KEY ## _config_trait::key )

// Get value from config blockusing trait
#define reconfig_value_using_trait(CONF,KEY) CONF->get_value< KEY ## _config_trait::type >( KEY ## _config_trait::key )


/**
 * \brief Create type trait.
 *
 * A type trait is used to bind a local trait name to a system wide
 * canonical type name string to a c++ type name. This is useful way
 * to establish names for types that are used throughout a sprokit
 * pipeline.
 *
 * Type traits should name a logical or semantic type not a physical
 * or logical type. This essential for verifying the semantics of a
 * pipeline. For example, GSD is usually a double but the trait name
 * should be \b gsd with a type double. It is a really bad idea to
 * name type traits based on the concrete builtin fundamental type
 * such as double or int.
 *
 \code
 create_type_trait( gsd, "kwiver:gsd", double );  // do this
 create_type_trait( double, "kwiver:double", double );  // DO NOT DO THIS !!!
 \endcode
 *
 * The canonical type name is a string that will be used to identify
 * the type and is used to verify the compatibility of process ports
 * when making connections. Only ports with the same canonical type
 * name can be connected.
 *
 * For small systems, these names can specify the logical data item
 * passing through the ports. For larger systems, it may make sense to
 * establish a hierarchical name space. One way to do this is to
 * separate the levels with a ':' character as shown in the
 * examples. Using qualified names reduces the change of name
 * collisions when two subsystems pick the same logical name for
 * different underlying types.
 *
 * Examples of defining type traits
 \code
 create_type_trait( image, "kwiver:image_container", kwiver::vital::image_container_sptr ); // polymorphic type must pass by reference
 \endcode
 *
 * This type trait name is used when defining port traits ( \ref create_port_trait() ).
 *
 * \param TN Trait type name
 * \param CTN Canonical type name.
 * \param TYPE C++ concrete type name (e.g. std::string )
 */
#define create_type_trait(TN, CTN, TYPE)                                \
namespace { struct TN ## _type_trait {                                  \
  static const sprokit::process::type_t name;                           \
  typedef TYPE type;                                                    \
};                                                                      \
sprokit::process::type_t const TN ## _type_trait::name = sprokit::process::type_t( CTN ); }


/**
 * \brief Create named port trait.
 *
 * A port trait is used to define all properties of a sprokit process
 * input or output port. The main purpose of port traits is to
 * centralize the definition of port attributes that are needed for
 * the various port operations such as declaring a port and
 * transferring data through the port.
 *
 * A sprokit process port is defined using the
 * declare_input_port_using_trait() and
 * declare_output_port_using_trait().
 *
 * Port traits are defined as follows. Sometimes the trait name and
 * the data type are the same, so don't get confused by this.
 *
 \code
 create_port_trait( timestamp, timestamp, "Timestamp for input image." );
 create_port_trait( src_to_ref_homography, homography_src_to_ref, "Source image to ref image homography." );
 \endcode
 *
 * \param PN Sprokit process port name
 * \param TN Type trait name as defined with a create_type_trait()
 * \param DESCRIP Port description
 *
 * \sa declare_input_port_using_trait()
 * \sa declare_output_port_using_trait()
 */
#define create_port_trait(PN, TN, DESCRIP)                              \
  namespace { struct PN ## _port_trait {                                \
  static const sprokit::process::type_t             type_name;          \
  static const sprokit::process::port_t             port_name;          \
  static const sprokit::process::port_description_t description;        \
  typedef TN ## _type_trait::type type;                                 \
};                                                                      \
sprokit::process::type_t const PN ## _port_trait::type_name = sprokit::process::type_t( TN ## _type_trait::name ); \
sprokit::process::port_t const PN ## _port_trait::port_name = sprokit::process::port_t( # PN ); \
sprokit::process::port_description_t const PN ## _port_trait::description = sprokit::process::port_description_t( DESCRIP ); }

#if VITAL_VARIADAC_MACRO

//
// Substantial macro magic
//
#define DPFT4( D, PN, FLAG, FREQ )                      \
declare_ ## D ## _port( PN ## _port_trait::port_name,   \
                        PN ## _port_trait::type_name,   \
                        FLAG,                           \
                        PN ## _port_trait::description, \
                        FREQ )

#define DPFT5( D, PN, FLAG, FREQ, DESCRIP )             \
declare_ ## D ## _port( PN ## _port_trait::port_name,   \
                        PN ## _port_trait::type_name,   \
                        FLAG,                           \
                        DESCRIP,                        \
                        FREQ )

#define DPUT3( D, PN, FLAG )         \
declare_ ## D ## _port( PN ## _port_trait::port_name,   \
                        PN ## _port_trait::type_name,   \
                        FLAG,                           \
                        PN ## _port_trait::description)

#define DPUT4( D, PN, FLAG, DESCRIP  )                  \
declare_ ## D ## _port( PN ## _port_trait::port_name,   \
                        PN ## _port_trait::type_name,   \
                        FLAG,                           \
                        DESCRIP)

#define GET_MACRO(_1,_2,_3,_4,NAME, ...) NAME


/**
 * \brief Declare sprokit input port using a port trait.
 *
 * This macro is used to declare a sprokit input port to the pipeline
 * framework based on the specified port trait.
 *
 \code
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( timestamp, required, "description" );
 \endcode
 *
 * \param PN Port trait name as defined by create_port_trait()
 * \param FLAG Port flags as defined by sprokit::process::port_flags_t
 * \param DESCRIP Optional port description
 */
#define declare_input_port_using_trait(...) \
  GET_MACRO(__VA_ARGS__, xxx, DPUT4, DPUT3)(input, __VA_ARGS__)


/**
 * \brief Declare sprokit output port using port trait.
 *
 * This macro is used to declare a sprokit output port to the pipeline
 * framework based on the specified port trait.
 *
 \code
  sprokit::process::port_flags_t optional;

  declare_output_port_using_trait( src_to_ref_homography, optional );
  declare_output_port_using_trait( src_to_ref_homography, optional, "description" );
 \endcode
 *
 * \param PN Port trait name as defined by create_port_trait()
 * \param FLAG Port flags as defined by sprokit::process::port_flags_t
 * \param DESCRIP Optional port description
 */
#define declare_output_port_using_trait(...) \
  GET_MACRO(__VA_ARGS__, xxx, DPUT4, DPUT3)(output, __VA_ARGS__)


/**
 * \brief Declare sprokit input port using a port trait.
 *
 * This macro is used to declare a sprokit input port with a frequency
 * to the pipeline framework based on the specified port trait.
 *
 \code
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_input_port_with_freq_using_trait( timestamp, required );
  declare_input_port_with_freq_using_trait( timestamp, required, "description" );
 \endcode
 *
 * \param PN Port trait name as defined by create_port_trait()
 * \param FLAG Port flags as defined by sprokit::process::port_flags_t
 * \param FREQ Port frequency
 * \param DESCRIP Optional port description
 */
#define declare_input_port_with_freq_using_trait(...) \
  GET_MACRO(__VA_ARGS__, DPFT5, DPFT4, xxx)(input, __VA_ARGS__)


/**
 * \brief Declare sprokit output port using port trait.
 *
 * This macro is used to declare a sprokit output with a frequency
 * specification port to the pipeline framework based on the specified
 * port trait.
 *
 \code
  sprokit::process::port_flags_t optional;

  declare_output_port_with_freq_using_trait( src_to_ref_homography, optional );
  declare_output_port_with_freq_using_trait( src_to_ref_homography, optional, "description" );
 \endcode
 *
 * \param PN Port trait name as defined by create_port_trait()
 * \param FLAG Port flags as defined by sprokit::process::port_flags_t
 * \param FREQ Port frequency
 * \param DESCRIP Optional port description
 */
#define declare_output_port_with_freq_using_trait(...) \
  GET_MACRO(__VA_ARGS__, DPFT5, DPFT4, xxx)(output, __VA_ARGS__)

#else

//
// Some compilers have trouble with the preceding approach.
//

#define declare_port_using_trait( D, PN, FLAG, ... )    \
declare_ ## D ## _port( PN ## _port_trait::port_name,   \
                        PN ## _port_trait::type_name,   \
                        FLAG,                           \
                        PN ## _port_trait::description)

#define declare_port_with_freq_using_trait( D, PN, FLAG, ... )  \
declare_ ## D ## _port( PN ## _port_trait::port_name,           \
                        PN ## _port_trait::type_name,           \
                        FLAG,                                   \
                        PN ## _port_trait::description)

#define declare_input_port_using_trait( PN, FLAG, ... ) declare_port_using_trait( input, PN, FLAG, __VA_ARGS__ )
#define declare_output_port_using_trait( PN, FLAG, ... ) declare_port_using_trait( output, PN, FLAG, __VA_ARGS__ )

#define declare_input_port_with_freq_using_trait( PN, FLAG, FREQ, ... ) declare_port_with_freq_using_trait( input, PN, FLAG, __VA_ARGS__ )

#define declare_output_port_with_freq_using_trait( PN, FLAG, FREQ, ... ) declare_port_with_freq_using_trait( output, PN, FLAG, __VA_ARGS__ )

#endif

/**
 * \brief Get input from port using port trait name.
 *
 * This macro returns a data value form a port specified by the port
 * trait or the configured static value. If there is a value on the
 * port, then this method behaves the same as grab_from_port_using_trait().
 *
 * This should be used with ports that have the \c flag_input_static
 * option set when created.
 *
 \code
 create_type_trait( timestamp, "kwiver:timestamp", kwiver::vital::timestamp );
 kwiver::vital::timestamp frame_time = grab_input_using_trait( timestamp );
 \endcode
 *
 * \sa sprokit::process::grab_input_as()
 *
 * \param PN Port trait name.
 *
 * \return Data value from port or default value from config.
 */
#define grab_input_using_trait(PN)                                      \
grab_input_as< PN ## _port_trait::type > ( PN ## _port_trait::port_name )


/**
 * \brief Get input from port using port trait name.
 *
 * This method grabs an input value directly from the port specified
 * by the port trait with \b no handling for static ports. This call will
 * block until a datum is available.
 *
 \code
 create_type_trait( timestamp, "kwiver:timestamp", kwiver::vital::timestamp );
 kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );
 \endcode
 *
 * Optional ports can be handled as follows:
 *
\code
  // See if optional input port has been connected.
  if ( has_input_port_edge_using_trait( timestamp ) )
  {
    frame_time = grab_input_using_trait( timestamp );
  }
\endcode
 *
 * \sa sprokit::process::grab_from_port_as()
 *
 * \param PN Port trait name.
 *
 * \return Data value from port.
 */
#define grab_from_port_using_trait(PN)                                  \
grab_from_port_as< PN ## _port_trait::type > ( PN ## _port_trait::port_name )



/**
 * \brief Get input from port using port trait name.
 *
 * Grab a datum packet from port specified by the port trait.
 * The datum packet contains the port data and other metadata.
 * See \ref sprokit::datum for details.
 *
 * \param PN Port trait name.
 *
 * \return The datum available on the port.
 */
#define grab_datum_from_port_using_trait(PN)            \
  grab_datum_from_port( PN ## _port_trait::port_name )


/**
 * \brief Get edge datum from port using port trait name.
 *
 * Grab a edge datum packet from port specified by the port trait.
 * The edge datum packet contains the raw data and metadata.  See
 * \ref sprokit::edge_datum_t for details.
 *
 * \param PN Port trait name.
 *
 * \return The edge datum available on the port.
 */
#define grab_edge_datum_using_trait(PN)            \
  grab_from_port( PN ## _port_trait::port_name )


/**
 * \brief Peek at a edge packet from a port.
 *
 * This macro peeks at the first edge packet available on the port
 * specified by the port trait.
 *
 * \param PN Port trait name.
 *
 * \return The edge from the port queue.
 */
#define peek_at_port_using_trait(PN)                   \
  peek_at_port(PN ## _port_trait::port_name)


/**
 * \brief Peek at a datum packet from a port.
 *
 * This macro peeks at the first datum packet available on the port
 * specified by the port trait.
 *
 * \param PN Port trait name.
 *
 * \return The datum from the port queue.
 */
#define peek_at_datum_using_trait(PN)                   \
  peek_at_datum_on_port(PN ## _port_trait::port_name)


/**
 * \brief Peek at a datum packet from a port.
 *
 * This macro peeks at a datum packet available on the port specified
 * by the port trait.
 *
 * \param PN Port trait name.
 * \param IDX The element within the port queue to look at. Defaults
 * to zero (the top element)
 *
 * \return The datum from the port queue.
 */
#define peek_at_datum_n_using_trait(PN, IDX)                    \
  peek_at_datum_on_port(PN ## _port_trait::port_name, IDX)


/**
 * \brief Test to see if port is connected.
 *
 * This macro tests to see if the specified port is connected.
 *
 * \param PN Port trait name.
 *
 * \return \b true if port is connected
 */
#define has_input_port_edge_using_trait(PN)             \
  has_input_port_edge(PN ##_port_trait::port_name )


/**
 * \brief Count number of edged connected to output port.
 *
 * This method returns the number of down stream processes are
 * connected to the port. This is useful in determining if there
 * are any consumers of an output port.
 *
 * For example, this can be used to optimize a process. An
 * expensive output can be skipped if there are no consumers.
 *
 * \param PN Port trait name.
 *
 * \returns The number of edges connected to the \p port.
 */

#define count_output_port_edges_using_trait(PN) \
  count_output_port_edges(PN ##_port_trait::port_name )



// Putting data to ports

/**
 * \brief Push data value to port.
 *
 * Push data value to port specified in port trait.
 *
 * \param PN Port trait name.
 * \param VAL Data value to put to port.
 *
 * @return
 */
#define push_to_port_using_trait(PN, VAL)                               \
push_to_port_as< PN ## _port_trait::type > ( PN ## _port_trait::port_name, VAL )


/**
 * \brief Push port datum value to port.
 *
 * Push port datum structure to port specified in port trait.
 *
 * See \ref sprokit::datum for details.
 *
 * \param PN Port trait name.
 * \param VAL Port datum value to send to port.
 *
 * \return
 */
#define push_datum_to_port_using_trait(PN,VAL)          \
push_datum_to_port( PN ## _port_trait::port_name, VAL )

#endif /* _KWIVER_TRAIT_UTILS_H_ */
