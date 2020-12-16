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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

/**
 * \file
 * \brief C Interface to \p config_block object implementation
 */


#include "config_block.h"

#include <cstdlib>
#include <cstring>

#include <map>
#include <sstream>
#include <string>

#include <vital/config/config_block.h>
#include <vital/config/config_block_io.h>
#include <vital/config/config_block_exception.h>
#include <vital/logger/logger.h>
#include <vital/exceptions.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/config_block.h>

namespace kwiver {
namespace vital_c {

SharedPointerCache< kwiver::vital::config_block, vital_config_block_t >
CONFIG_BLOCK_SPTR_CACHE( "config_block" );

} }


// Static Constant getters

/// Separator between blocks within the config
char const*
vital_config_block_block_sep()
{
  static char const* static_bs = 0;

  if ( ! static_bs )
  {
    STANDARD_CATCH(
      "C::config_block::block_sep()", NULL,
      std::string bs( kwiver::vital::config_block::block_sep() );
      static_bs = bs.c_str();
    );
  }
  return static_bs;
}


/// The magic group for global parameters
char const*
vital_config_block_global_value()
{
  static char const* static_gv = 0;

  if ( ! static_gv )
  {
    STANDARD_CATCH(
      "C::config_block::global_value", NULL,
      std::string gv( kwiver::vital::config_block::global_value() );
      static_gv = gv.c_str();
    );
  }
  return static_gv;
}


/// Create a new, empty \p config_block object
vital_config_block_t*
vital_config_block_new()
{
  STANDARD_CATCH(
    "C::config_block::new", 0,

    return vital_config_block_new_named( "" );

  );
  return 0;
}


/// Create a new, empty \p config_block object with a name
vital_config_block_t*
vital_config_block_new_named( char const* name )
{
  STANDARD_CATCH(
    "C::config_block::new_named", 0,
    auto cb_sptr = kwiver::vital::config_block::empty_config( name );
    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.store( cb_sptr );
    return reinterpret_cast< vital_config_block_t* >( cb_sptr.get() );
  );
  return 0;
}


/// Destroy a config block object
void
vital_config_block_destroy( vital_config_block_t* cb,
                            vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::destroy", eh,
    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.erase( cb );
  );
}


/// Get the name of the \p config_block instance
char const*
vital_config_block_get_name( vital_config_block_t* cb,
                             vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::get_name", eh,
    std::string name =
      kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )->get_name();
    return name.c_str();
  );
  return 0;
}


/// Get a copy of a sub-block of the configuration
vital_config_block_t*
vital_config_block_subblock( vital_config_block_t* cb,
                             char const*           key,
                             vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::subblock", eh,
    auto cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    auto sb_sptr = cb_sptr->subblock( key );
    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.store( sb_sptr );
    return reinterpret_cast< vital_config_block_t* > ( sb_sptr.get() );
  );
  return 0;
}


/// Get a mutable view of a sub-block within a configuration
vital_config_block_t*
vital_config_block_subblock_view( vital_config_block_t* cb,
                                  char const*           key,
                                  vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::subblock_view", eh,
    auto cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    auto sb_sptr = cb_sptr->subblock_view( key );
    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.store( sb_sptr );
    return reinterpret_cast< vital_config_block_t* > ( sb_sptr.get() );
  );
  return 0;
}


/// Get the string value for a key
char const*
vital_config_block_get_value( vital_config_block_t* cb,
                              char const*           key,
                              vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::get_value", eh,
    auto cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    try
    {
      std::string v = cb_sptr->get_value< std::string >( key );
      char *vc = (char*)malloc( sizeof(char) * (v.size() + 1) );
      strcpy( vc, v.c_str() );
      return vc;
    }
    catch( kwiver::vital::no_such_configuration_value_exception const &e )
    {
      POPULATE_EH( eh, 1, e.what() );
    }
  );
  return 0;
}


/// Get the boolean value for a key
bool
vital_config_block_get_value_bool( vital_config_block_t*  cb,
                                   char const*            key,
                                   vital_error_handle_t*  eh )
{
  STANDARD_CATCH(
    "C::config_block:get_value_bool", eh,
    auto cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    try
    {
      return cb_sptr->get_value< bool >( key );
    }
    catch( kwiver::vital::no_such_configuration_value_exception const &e )
    {
      POPULATE_EH( eh, 1, e.what() );
    }
  );
  return false;
}


/// Get the string value for a key if it exists, else the default
char const*
vital_config_block_get_value_default( vital_config_block_t* cb,
                                      char const*           key,
                                      char const*           deflt,
                                      vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::get_value_default", eh,
    auto cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    std::string v = cb_sptr->get_value< std::string >( key, deflt );
    char *vc = (char*)malloc( sizeof(char) * (v.size() + 1) );
    strcpy( vc, v.c_str() );
    return vc;
  );
  return 0;
}


/// Get the boolean value for a key if it exists, else the default
bool
vital_config_block_get_value_default_bool( vital_config_block_t*  cb,
                                           char const*            key,
                                           bool                   deflt,
                                           vital_error_handle_t*  eh )
{
  STANDARD_CATCH(
    "C::config_block::get_value_default_bool", eh,
    return kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )
      ->get_value< bool > ( key, deflt );
  );
  return false;
}


/// Get the description string for a given key
char const*
vital_config_block_get_description( vital_config_block_t* cb,
                                    char const*           key,
                                    vital_error_handle_t* eh)
{
  STANDARD_CATCH(
    "C::config_block::get_description", eh,
    auto cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    try
    {
      std::string d = cb_sptr->get_description( key );
      char *dc = (char*)malloc( sizeof(char) * (d.size() + 1) );
      strcpy( dc, d.c_str() );
      return dc;
    }
    catch( kwiver::vital::no_such_configuration_value_exception const& e )
    {
      POPULATE_EH( eh, 1, e.what() );
    }
  );
  return 0;
}


/// Set the string value for a key
void
vital_config_block_set_value( vital_config_block_t* cb,
                              char const*           key,
                              char const*           value,
                              vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::set_value", eh,
    auto cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    try
    {
      cb_sptr->set_value< std::string >( key, value );
    }
    catch( kwiver::vital::set_on_read_only_value_exception const& e )
    {
      POPULATE_EH( eh, 1, e.what() );
    }
  );
}


/// Set a string value with an associated description
void
vital_config_block_set_value_descr( vital_config_block_t* cb,
                                    char const*           key,
                                    char const*           value,
                                    char const*           description,
                                    vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::set_value_descr", eh,
    auto cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    try
    {
      cb_sptr->set_value< std::string >( key, value, description );
    }
    catch( kwiver::vital::set_on_read_only_value_exception const& e )
    {
      POPULATE_EH( eh, 1, e.what() );
    }
  );
}


/// Remove a key/value pair from the configuration.
void
vital_config_block_unset_value( vital_config_block_t* cb,
                                char const*           key,
                                vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::unset_value", eh,
    try
    {
      kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )->unset_value( key );
    }
    catch( kwiver::vital::unset_on_read_only_value_exception const& e )
    {
      POPULATE_EH( eh, 1, e.what() );
    }
    catch( kwiver::vital::no_such_configuration_value_exception const& e )
    {
      POPULATE_EH( eh, 2, e.what() );
    }
  );
}


/// Query if a value is read-only
bool
vital_config_block_is_read_only( vital_config_block_t* cb,
                                 char const*           key,
                                 vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C:config_block::is_read_only", eh,
    return kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )->is_read_only( key );
  );
  return false;
}


/// Mark the given key as read-only
void
vital_config_block_mark_read_only( vital_config_block_t* cb,
                                   char const*           key,
                                   vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::mark_read_only", eh,
    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )->mark_read_only( key );
  );
}


/// Merge another \p config_block's entries into this \p config_block
void
vital_config_block_merge_config( vital_config_block_t*  cb,
                                 vital_config_block_t*  other,
                                 vital_error_handle_t*  eh )
{
  STANDARD_CATCH(
    "C::config_block::merge_config", eh,
    auto cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    auto other_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( other );
    try
    {
      cb_sptr->merge_config( other_sptr );
    }
    catch( kwiver::vital::set_on_read_only_value_exception const& e )
    {
      POPULATE_EH( eh, 1, e.what() );
    }
  );
}


/// Check if a value exists for the given key
bool
vital_config_block_has_value( vital_config_block_t* cb,
                              char const*           key,
                              vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::has_key", eh,
    return kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )->has_value( key );
  );
  return false;
}


/// Return the values available in the configuration.
void
vital_config_block_available_values( vital_config_block_t*  cb,
                                     unsigned int*          length,
                                     char***                keys,
                                     vital_error_handle_t*  eh )
{
  STANDARD_CATCH(
    "C::config_block::available_values", eh,
    if ( ( length == 0 ) || ( keys == 0 ) )
    {
      throw kwiver::vital::invalid_value( "One or both provided output parameters "
                                          "were a NULL pointer." );
    }

    auto cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    std::vector< std::string > cb_keys = cb_sptr->available_values();
    kwiver::vital_c::make_string_list( cb_keys, *length, *keys );
  );
}


namespace {

// Helper to read a config_block from a file
template <typename... Args>
vital_config_block_t*
read_config_file_helper( vital_error_handle_t* eh, Args... args )
{
  try
  {
    auto c = kwiver::vital::read_config_file( args... );
    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.store( c );
    return reinterpret_cast< vital_config_block_t* > ( c.get() );
  }
  catch ( kwiver::vital::config_file_not_found_exception const& e )
  {
    POPULATE_EH( eh, 1, e.what() );
  }
  catch ( kwiver::vital::config_file_not_read_exception const& e )
  {
    POPULATE_EH( eh, 2, e.what() );
  }
  catch ( kwiver::vital::config_file_not_parsed_exception const& e )
  {
    POPULATE_EH( eh, 3, e.what() );
  }

  return 0;
}

}


/// Read in a configuration file, producing a config_block object
vital_config_block_t*
vital_config_block_file_read( char const*           filepath,
                              vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::file_read", eh,
    // TODO: Reflect full C++ function sig in C function sig
    return read_config_file_helper( eh,
                                    filepath,
                                    kwiver::vital::config_path_list_t{},
                                    false );
  );
  return 0;
}


/// Read in a configuration file, producing a named config_block object
vital_config_block_t*
vital_config_block_file_read_from_standard_location(
  char const*           name,
  char const*           application_name,
  char const*           application_version,
  char const*           install_prefix,
  bool                  merge,
  vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "C::config_block::file_read_first_from_standard_location", eh,
    return read_config_file_helper( eh,
                                    name,
                                    MAYBE_EMPTY_STRING( application_name ),
                                    MAYBE_EMPTY_STRING( application_version ),
                                    MAYBE_EMPTY_STRING( install_prefix ),
                                    merge );
  );
  return 0;
}


/// Output to file the given \c config_block object to the specified file path
void
vital_config_block_file_write( vital_config_block_t*  cb,
                               char const*            filepath,
                               vital_error_handle_t*  eh )
{
  STANDARD_CATCH(
    "C::config_block::file_write", eh,
    try
    {
      kwiver::vital::config_block_sptr c = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
      kwiver::vital::write_config_file( c, filepath );
    }
    catch ( kwiver::vital::config_file_write_exception const& e )
    {
      POPULATE_EH( eh, 1, e.what() );
    }
  );
}
