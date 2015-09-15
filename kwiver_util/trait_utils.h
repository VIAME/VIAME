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

#include <kwiver/vital/config/config_block_types.h>
#include <sprokit/pipeline/process.h>

//
// These macros are designed to assist in establishing a system wide
// set of consistent types, config keys, and ports.
// Create trait once, use many times.
//

#define create_config_trait(KEY, TYPE, DEF, DESCR)                      \
namespace  { struct KEY ## _config_trait {                              \
  static const kwiver::vital::config_block_key_t      key;              \
  static const kwiver::vital::config_block_value_t    def;              \
  static const kwiver::vital::config_block_description_t description;   \
  typedef TYPE type;                                                    \
};                                                                      \
kwiver::vital::config_block_key_t const KEY ## _config_trait::key = kwiver::vital::config_block_key_t( # KEY ); \
kwiver::vital::config_block_value_t const KEY ## _config_trait::def = kwiver::vital::config_block_value_t( DEF ); \
kwiver::vital::config_block_description_t const  KEY ## _config_trait::description = kwiver::vital::config_block_description_t( DESCR ); }

#define declare_config_using_trait(KEY)                         \
declare_configuration_key( KEY ## _config_trait::key,           \
                           KEY ## _config_trait::def,           \
                           KEY ## _config_trait::description)

// Get value from config using trait
#define config_value_using_trait(KEY) config_value< KEY ## _config_trait::type >( KEY ## _config_trait::key )


// Type trait consists of canonical type name and concrete type
// ( type-trait-name, "canonical-type-name", concrete-type )
#define create_type_trait( TN, CTN, TYPE)                                \
namespace { struct TN ## _type_trait {                                  \
  static const sprokit::process::type_t name;                           \
  typedef TYPE type;                                                    \
};                                                                      \
sprokit::process::type_t const TN ## _type_trait::name = sprokit::process::type_t( CTN ); }

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
#define grab_input_using_trait(PN) \
grab_input_as< PN ## _port_trait::type > ( PN ## _port_trait::port_name )

#define grab_from_port_using_trait(PN) \
grab_from_port_as< PN ## _port_trait::type > ( PN ## _port_trait::port_name )

#define grab_datum_from_port_using_trait(PN) \
  grab_datum_from_port( PN ## _port_trait::port_name )

// Putting data to ports
#define push_to_port_using_trait(PN, VAL) \
push_to_port_as< PN ## _port_trait::type > ( PN ## _port_trait::port_name, VAL )

#define push_datum_to_port_using_trait(PN,VAL) \
push_datum_to_port( PN ## _port_trait::port_name, VAL )

#endif /* _KWIVER_TRAIT_UTILS_H_ */
