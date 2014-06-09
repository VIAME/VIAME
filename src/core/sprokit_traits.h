/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_SPROKIT_TRAITS_H_
#define _KWIVER_SPROKIT_TRAITS_H_

#include <sprokit/pipeline/process.h>

//
// These macros are designed to assist in establishing a system wide
// set of consistent types, config keys, and ports.
// Create trait once, use many times.
//

#define create_config_trait(KEY, TYPE, DEF, DESCR)                      \
namespace  { struct KEY ## _config_trait {                              \
  static const sprokit::config::key_t      key;                         \
  static const sprokit::config::value_t    def;                         \
  static const sprokit::config::description_t description;              \
  typedef TYPE type;                                                    \
};                                                                      \
sprokit::config::key_t const KEY ## _config_trait::key = sprokit::config::key_t( # KEY ); \
sprokit::config::value_t const KEY ## _config_trait::def = sprokit::config::value_t( DEF ); \
sprokit::config::description_t const  KEY ## _config_trait::description = sprokit::config::description_t( DESCR ); }

#define declare_config_using_trait(KEY)                         \
declare_configuration_key( KEY ## _config_trait::key,           \
                           KEY ## _config_trait::def,           \
                           KEY ## _config_trait::description)

// Get value from config using trait
#define config_value_using_trait(KEY) config_value< KEY ## _config_trait::type >( KEY ## _config_trait::key )


// Type trait consists of canonical type name and underlying type
#define create_type_trait( TN, TYPE)                                    \
namespace { struct TN ## _type_trait {                                  \
  static const sprokit::process::type_t name;                           \
  typedef TYPE type;                                                    \
};                                                                      \
sprokit::process::type_t const TN ## _type_trait::name = sprokit::process::type_t( # TN ); }

//+  std::istream& operator<< (std::ostream& str, TN ## _type_trait::type) // optionally declare input operator


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


#define declare_port_using_trait( D, PN, FLAG )         \
declare_ ## D ## _port( PN ## _port_trait::port_name,   \
                        PN ## _port_trait::type_name,   \
                        FLAG,                           \
                        PN ## _port_trait::description)

#define declare_input_port_using_trait( PN, FLAG ) declare_port_using_trait( input, PN, FLAG )
#define declare_output_port_using_trait( PN, FLAG ) declare_port_using_trait( output, PN, FLAG )



              /* Would be nice to select call based on presence of input operator
#define grab_input_using_trait(PN)
#if PN ## _HAS_INPUT_OPERATOR
grab_input_as< PN ## _port_trait::type > ( PN ## _port_trait::port_name )
#else
grab_from_port_as< PN ## _port_trait::type > ( PN ## _port_trait::port_name )
#endif
              */

// Getting data from ports
#define grab_input_using_trait(PN) grab_input_as< PN ## _port_trait::type > ( PN ## _port_trait::port_name )
#define grab_from_port_using_trait(PN) grab_from_port_as< PN ## _port_trait::type > ( PN ## _port_trait::port_name )

#endif /* _KWIVER_SPROKIT_TRAITS_H_ */
