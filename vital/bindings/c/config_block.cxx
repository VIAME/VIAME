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

#include <kwiver_util/config/config_block.h>
#include <kwiver_util/config/config_block_io.h>
#include <kwiver_util/config/config_block_exception.h>
#include <kwiver_util/logger/logger.h>
#include <vital/exceptions.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/config_block.h>


/// Definition of sptr cache
namespace kwiver {
namespace vital_c {

SharedPointerCache< kwiver::config_block, vital_config_block_t >
  CONFIG_BLOCK_SPTR_CACHE( "config_block" );

} }


// Static Constant getters

/// Separator between blocks within the config
vital_string_t* vital_config_block_block_sep()
{
  static vital_string_t *static_bs = 0;
  if( ! static_bs )
  {
    STANDARD_CATCH(
      "C::config_block::block_sep", NULL,
      std::string bs( kwiver::config_block::block_sep );
      static_bs = vital_string_new( bs.size(), bs.c_str() );
    );
  }
  return static_bs;
}

/// The magic group for global parameters
vital_string_t* vital_config_block_global_value()
{
  static vital_string_t *static_gv = 0;
  if( ! static_gv )
  {
    STANDARD_CATCH(
      "C::config_block::global_value", NULL,
      std::string gv( kwiver::config_block::global_value );
      static_gv = vital_string_new( gv.size(), gv.c_str() );
    );
  }
  return static_gv;
}


/// Create a new, empty \p config_block object
vital_config_block_t* vital_config_block_new()
{
  STANDARD_CATCH(
    "C::config_block::new", 0,

    return vital_config_block_new_named("");

  );
  return 0;
}

/// Create a new, empty \p config_block object with a name
vital_config_block_t* vital_config_block_new_named( char const *name )
{
  STANDARD_CATCH(
    "C::config_block::new_named", 0,

    kwiver::config_block_sptr cb_sptr = kwiver::config_block::empty_config( name );
    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.store( cb_sptr );
    return reinterpret_cast<vital_config_block_t*>(cb_sptr.get());
  );
  return 0;
}

/// Destroy a config block object
void vital_config_block_destroy( vital_config_block_t *cb,
                                 vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::config_block::destroy", eh,
    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.erase( cb );
  );
}

/// Get the name of the \p config_block instance
vital_string_t* vital_config_block_get_name( vital_config_block_t *cb )
{
  STANDARD_CATCH(
    "C::config_block::get_name", 0,
    std::string name = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )->get_name();
    return vital_string_new(name.length(), name.c_str());
  );
  return 0;
}

/// Get a copy of a sub-block of the configuration
vital_config_block_t* vital_config_block_subblock( vital_config_block_t *cb,
                                                   char const *key )
{
  STANDARD_CATCH(
    "C::config_block::subblock", 0,

    kwiver::config_block_sptr cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    kwiver::config_block_sptr sb_sptr = cb_sptr->subblock( key );
    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.store( sb_sptr );
    return reinterpret_cast<vital_config_block_t*>( sb_sptr.get() );
  );
  return 0;
}

/// Get a mutable view of a sub-block within a configuration
vital_config_block_t* vital_config_block_subblock_view( vital_config_block_t *cb,
                                                        char const *key )
{
  STANDARD_CATCH(
    "C::config_block::subblock_view", 0,

    kwiver::config_block_sptr cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    kwiver::config_block_sptr sb_sptr = cb_sptr->subblock_view( key );
    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.store( sb_sptr );
    return reinterpret_cast<vital_config_block_t*>( sb_sptr.get() );
  );
  return 0;
}

/// Get the string value for a key
vital_string_t* vital_config_block_get_value( vital_config_block_t *cb,
                                              char const* key )
{
  STANDARD_CATCH(
    "C::config_block::get_value", 0,

    std::string v = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )
        ->get_value<std::string>( key );
    return vital_string_new( v.length(), v.c_str() );
  );
  return 0;
}

/// Get the boolean value for a key
bool vital_config_block_get_value_bool( vital_config_block_t *cb,
                                        char const *key,
                                        vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::config_block:get_value_bool", eh,
    return kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )
      ->get_value<bool>( key );
  );
  return false;
}

/// Get the string value for a key if it exists, else the default
vital_string_t*  vital_config_block_get_value_default( vital_config_block_t *cb,
                                                       char const* key,
                                                       char const* deflt )
{
  STANDARD_CATCH(
    "C::config_block::get_value_default", 0,

    std::string v = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )
        ->get_value<std::string>( key, deflt );
    return vital_string_new( v.length(), v.c_str() );
  );
  return 0;
}

/// Get the boolean value for a key if it exists, else the default
bool vital_config_block_get_value_default_bool( vital_config_block_t *cb,
                                                char const *key,
                                                bool deflt,
                                                vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::config_block::get_value_default_bool", eh,
    return kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )
      ->get_value<bool>( key, deflt );
  );
  return false;
}

/// Get the description string for a given key
vital_string_t* vital_config_block_get_description( vital_config_block_t *cb,
                                                    char const* key )
{
  STANDARD_CATCH(
    "C::config_block::get_description", 0,

    std::string d = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )
        ->get_description( key ).c_str();
    return vital_string_new( d.length(), d.c_str() );
  );
  return 0;
}

/// Set the string value for a key
void vital_config_block_set_value( vital_config_block_t *cb,
                                   char const* key,
                                   char const* value )
{
  STANDARD_CATCH(
    "C::config_block::set_value", 0,

    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )
        ->set_value<std::string>( key, value );
  );
}

/// Set a string value with an associated description
void vital_config_block_set_value_descr( vital_config_block_t *cb,
                                         char const *key,
                                         char const *value,
                                         char const *description )
{
  STANDARD_CATCH(
    "C::config_block::set_value_descr", 0,

    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )
        ->set_value<std::string>( key, value, description );
  );
}

/// Remove a key/value pair from the configuration.
void vital_config_block_unset_value( vital_config_block_t *cb,
                                     char const *key )
{
  STANDARD_CATCH(
    "C::config_block::unset_value", 0,

    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )->unset_value( key );
  );
}

/// Query if a value is read-only
bool vital_config_block_is_read_only( vital_config_block_t *cb,
                                      char const *key )
{
  STANDARD_CATCH(
    "C:config_block::is_read_only", 0,

    return kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )->is_read_only( key );
  );
  return false;
}

/// Mark the given key as read-only
void vital_config_block_mark_read_only( vital_config_block_t *cb,
                                        char const *key )
{
  STANDARD_CATCH(
    "C::config_block::mark_read_only", 0,

    kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )->mark_read_only( key );
  );
}

/// Merge another \p config_block's entries into this \p config_block
void vital_config_block_merge_config( vital_config_block_t *cb,
                                      vital_config_block_t *other )
{
  STANDARD_CATCH(
    "C::config_block::merge_config", 0,

    kwiver::config_block_sptr cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    kwiver::config_block_sptr other_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( other );
    cb_sptr->merge_config( other_sptr );
  );
}

/// Check if a value exists for the given key
bool vital_config_block_has_value( vital_config_block_t *cb,
                                   char const *key )
{
  STANDARD_CATCH(
    "C::config_block::has_key", 0,

    return kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )->has_value( key );
  );
  return false;
}

/// Return the values available in the configuration.
void vital_config_block_available_values( vital_config_block_t *cb,
                                          unsigned int *length,
                                          char ***keys )
{
  STANDARD_CATCH(
    "C::config_block::available_values", 0,

    if ( length == 0 || keys == 0 )
    {
      throw kwiver::vital::invalid_value("One or both provided output parameters "
                                 "were a NULL pointer.");
    }

    kwiver::config_block_sptr cb_sptr = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
    std::vector<std::string> cb_keys = cb_sptr->available_values();
    kwiver::vital_c::make_string_list( cb_keys, *length, *keys );
  );
}


/// Read in a configuration file, producing a config_block object
vital_config_block_t* vital_config_block_file_read( char const *filepath,
                                                    vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::config_block::file_read", eh,
    try
    {
      kwiver::config_block_sptr c = kwiver::read_config_file( filepath );
      kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.store( c );
      return reinterpret_cast<vital_config_block_t*>( c.get() );
    }
    catch( kwiver::file_not_found_exception const &e )
    {
      eh->error_code = 1;
      eh->message = (char*)malloc(sizeof(char)*strlen(e.what()));
      strcpy( eh->message, e.what() );
    }
    catch( kwiver::file_not_read_exception const &e )
    {
      eh->error_code = 2;
      eh->message = (char*)malloc(sizeof(char)*strlen(e.what()));
      strcpy( eh->message, e.what() );
    }
    catch( kwiver::file_not_parsed_exception const &e )
    {
      eh->error_code = 3;
      eh->message = (char*)malloc(sizeof(char)*strlen(e.what()));
      strcpy( eh->message, e.what() );
    }
    );
  return 0;
}


/// Read in a configuration file, producing a named config_block object
vital_config_block_t* vital_config_block_file_read_with_name( char const *filepath,
                                                              char const *blockname,
                                                              vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::config_block::file_read_with_name", eh,
    try
    {
      kwiver::config_block_sptr c = kwiver::read_config_file( filepath,
                                                            blockname );
      kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.store( c );
      return reinterpret_cast<vital_config_block_t*>( c.get() );
    }
    catch( kwiver::file_not_found_exception const &e )
    {
      eh->error_code = 1;
      eh->message = (char*)malloc(sizeof(char)*strlen(e.what()));
      strcpy( eh->message, e.what() );
    }
    catch( kwiver::file_not_read_exception const &e )
    {
      eh->error_code = 2;
      eh->message = (char*)malloc(sizeof(char)*strlen(e.what()));
      strcpy( eh->message, e.what() );
    }
    catch( kwiver::file_not_parsed_exception const &e )
    {
      eh->error_code = 3;
      eh->message = (char*)malloc(sizeof(char)*strlen(e.what()));
      strcpy( eh->message, e.what() );
    }
  );
  return 0;
}


/// Output to file the given \c config_block object to the specified file path
void vital_config_block_file_write( vital_config_block_t *cb,
                                    char const *filepath,
                                    vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::config_block::file_write", eh,
    try
    {
      kwiver::config_block_sptr c = kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb );
      kwiver::write_config_file( c, filepath );
    }
    catch( kwiver::file_write_exception const &e )
    {
      eh->error_code = 1;
      eh->message = (char*)malloc(sizeof(char)*strlen(e.what()));
      strcpy( eh->message, e.what() );
    }
  );
}
