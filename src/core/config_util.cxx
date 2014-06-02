/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "config_util.h"

#include <boost/foreach.hpp>


namespace kwiver
{

// ----------------------------------------------------------------
void convert_config( sprokit::config_t const  from,
                     maptk::config_block_sptr to )
{

  sprokit::config::keys_t from_keys = from->available_values();

  BOOST_FOREACH( sprokit::config::key_t key, from_keys )
  {
    if ( ! to->is_read_only( key ) )
    {
      sprokit::config::value_t const val = from->get_value< sprokit::config::value_t > ( key );
      to->set_value( key, val );

      // \todo add log message
      std::cout << "DEBUG - Processing entry: " << key << " = " << val << std::endl;

      // propagate read-only attribute
      if ( from->is_read_only( key ) )
      {
        to->mark_read_only( key );
      }
    }
    else
    {
      // \todo log warning - could not convert entry "key"
      std::cerr << "WARNING - could not convert read only entry " << key << std::endl;
    }
  } // end foreach
}


// ----------------------------------------------------------------
void convert_config( maptk::config_block_sptr const from,
                     sprokit::config_t              to )
{

  maptk::config_block_keys_t from_keys = from->available_values();

  BOOST_FOREACH( maptk::config_block_key_t key, from_keys )
  {
    if ( ! to->is_read_only( key ) )
    {
      std::string const val = from->get_value< std::string > ( key );

      // \todo add log message
      std::cout << "DEBUG - Processing entry: " << key << " = " << val << std::endl;

      to->set_value( key, val );

      // propagate read-only attribute
      if ( from->is_read_only( key ) )
      {
        to->mark_read_only( key );
      }
    }
    else
    {
      //+ log warning - could not convert entry "key"
      std::cerr << "WARNING - could not convert read only entry " << key << std::endl;
    }
  } // end foreach
}

} //end namespace
