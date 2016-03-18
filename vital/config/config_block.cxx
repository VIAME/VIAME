/*ckwg +29
 * Copyright 2011-2014 by Kitware, Inc.
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
 *
 * \brief Implementation of \link kwiver::vital::config_block configuration \endlink object
 */

#include "config_block.h"

#include <vital/vital_foreach.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <functional>
#include <cctype>
#include <locale>


namespace kwiver {
namespace vital {

namespace {

// trim from start
static inline std::string&
ltrim( std::string& s )
{
  s.erase( s.begin(), std::find_if( s.begin(), s.end(), std::not1( std::ptr_fun< int, int > ( std::isspace ) ) ) );
  return s;
}


// trim from end
static inline std::string&
rtrim( std::string& s )
{
  s.erase( std::find_if( s.rbegin(), s.rend(), std::not1( std::ptr_fun< int, int > ( std::isspace ) ) ).base(), s.end() );
  return s;
}


// trim from both ends
static inline std::string&
trim( std::string& s )
{
  return ltrim( rtrim( s ) );
}


bool starts_with( std::string const& str, std::string const& pfx )
{
  return ( str.substr( 0, pfx.size()) == pfx );
}

} // end namespace



config_block_key_t const config_block::block_sep = config_block_key_t( ":" );
config_block_key_t const config_block::global_value = config_block_key_t( "_global" );

/// private helper method for determining key path prefixes
static bool does_not_begin_with( config_block_key_t const&  key,
                                 config_block_key_t const&  name );

/// private helper method to strip a block name from a key path
static config_block_key_t strip_block_name( config_block_key_t const& subblock,
                                            config_block_key_t const& key );

// Create an empty configuration.
config_block_sptr
config_block
::empty_config( config_block_key_t const& name )
{
  // remember, config_block_sptr is a shared pointer
  // Create a new config block with no parent.
  return config_block_sptr( new config_block( name, config_block_sptr() ) );
}

// Destructor
config_block
::~config_block()
{
}


// ------------------------------------------------------------------
// Get the name of this \c config_block instance.
config_block_key_t
config_block
::get_name()
{
  return this->m_name;
}


// ------------------------------------------------------------------
// Get a subblock from the configuration.
config_block_sptr
config_block
::subblock( config_block_key_t const& key ) const
{
  config_block_sptr conf( new config_block( key, config_block_sptr() ) );

  VITAL_FOREACH( config_block_key_t const& key_name, available_values() )
  {
    if ( does_not_begin_with( key_name, key ) )
    {
      continue;
    }

    config_block_key_t const stripped_key_name = strip_block_name( key, key_name );

    conf->set_value( stripped_key_name,
                     i_get_value( key_name ),
                     get_description( key_name ) );
  }

  return conf;
}


// ------------------------------------------------------------------
// Get a subblock view into the configuration.
config_block_sptr
config_block
::subblock_view( config_block_key_t const& key )
{
  return config_block_sptr( new config_block( key, shared_from_this() ) );
}


// ------------------------------------------------------------------
config_block_description_t
config_block
::get_description( config_block_key_t const& key ) const
{
  if ( m_parent )
  {
    return m_parent->get_description( m_name + block_sep + key );
  }

  store_t::const_iterator i = m_descr_store.find( key );
  if ( i == m_descr_store.end() )
  {
    throw no_such_configuration_value_exception( key );
  }

  return i->second;
}


// ------------------------------------------------------------------
// Remove a value from the configuration.
void
config_block
::unset_value( config_block_key_t const& key )
{
  if ( m_parent )
  {
    m_parent->unset_value( m_name + block_sep + key );
  }
  else
  {
    if ( is_read_only( key ) )
    {
      config_block_value_t const current_value = get_value< config_block_value_t > ( key, config_block_value_t() );

      throw unset_on_read_only_value_exception( key, current_value );
    }

    store_t::iterator const i = m_store.find( key );
    store_t::iterator const j = m_descr_store.find( key );
    location_t::iterator const k = m_def_store.find( key );

    // value and descr stores managed in parallel, so if key doesn't exist in
    // value store, there will be no parallel value in the descr store.
    if ( i == m_store.end() )
    {
      throw no_such_configuration_value_exception( key );
    }

    m_store.erase( i );
    m_descr_store.erase( j );

    if ( k != m_def_store.end() )
    {
      m_def_store.erase( k );
    }
  }
}


// ------------------------------------------------------------------
// Query if a value is read-only.
bool
config_block
::is_read_only( config_block_key_t const& key ) const
{
  return 0 != m_ro_list.count( key );
}


// ------------------------------------------------------------------
// Set the value within the configuration as read-only.
void
config_block
::mark_read_only( config_block_key_t const& key )
{
  m_ro_list.insert( key );
}


// ------------------------------------------------------------------
// Merge the values in \p config_block into the current config.
void
config_block
::merge_config( config_block_sptr const& conf )
{
  config_block_keys_t const keys = conf->available_values();

  VITAL_FOREACH( config_block_key_t const & key, keys )
  {
    config_block_value_t const& val = conf->get_value< config_block_value_t > ( key );
    config_block_description_t const& descr = conf->get_description( key );

    i_set_value( key, val, descr );
  }
}


// ------------------------------------------------------------------
//Return the values available in the configuration.
config_block_keys_t
config_block
::available_values() const
{
  using namespace std::placeholders;  // for _1, _2, _3...
  config_block_keys_t keys;

  if ( m_parent )
  {
    config_block_keys_t parent_keys = m_parent->available_values();

    config_block_keys_t::iterator const i = std::remove_if( parent_keys.begin(), parent_keys.end(),
                                                            std::bind( does_not_begin_with, _1, m_name ) );

    parent_keys.erase( i, parent_keys.end() );

    std::transform( parent_keys.begin(), parent_keys.end(),
                    std::back_inserter( keys ), std::bind( strip_block_name, m_name, _1 ) );
  }
  else
  {
    VITAL_FOREACH( store_t::value_type const& value, m_store )
    {
      config_block_key_t const& key = value.first;

      keys.push_back( key );
    }
  }

  return keys;
}


// ------------------------------------------------------------------
// Check if a value exists for \p key.
bool
config_block
::has_value( config_block_key_t const& key ) const
{
  if ( m_parent )
  {
    return m_parent->has_value( m_name + block_sep + key );
  }

  return ( 0 != m_store.count( key ) );
}


// ------------------------------------------------------------------
// Internal constructor
config_block
::config_block( config_block_key_t const& name, config_block_sptr parent )
  : m_parent( parent ),
    m_name( name ),
    m_store(),
    m_descr_store(),
    m_ro_list(),
    m_def_store()
{
}


// ------------------------------------------------------------------
// private helper method to extract a value for a key
bool
config_block
::find_value( config_block_key_t const& key, config_block_value_t& val ) const
{
  if ( ! has_value( key ) )
  {
    return false;
  }

  val = i_get_value( key );
  return true;
}


// ------------------------------------------------------------------
// private value getter function
config_block_value_t
config_block
::i_get_value( config_block_key_t const& key ) const
{
  if ( m_parent )
  {
    return m_parent->i_get_value( m_name + block_sep + key );
  }

  store_t::const_iterator i = m_store.find( key );

  if ( i == m_store.end() )
  {
    return config_block_value_t();
  }

  return i->second;
}


// ------------------------------------------------------------------
// private key/value setter
void
config_block
::i_set_value( config_block_key_t const&          key,
               config_block_value_t const&        value,
               config_block_description_t const&  descr )
{
  if ( m_parent )
  {
    m_parent->set_value( m_name + block_sep + key, value, descr );
  }
  else
  {
    if ( is_read_only( key ) )
    {
      config_block_value_t const current_value = get_value< config_block_value_t > ( key, config_block_value_t() );

      throw set_on_read_only_value_exception( key, current_value, value );
    }

    config_block_value_t temp( value );
    m_store[key] = trim( temp ); // trim value in place. Leading and trailing blanks are evil!

    // Only assign the description given if there is no stored description
    // for this key, or the given description is non-zero.
    if ( ( m_descr_store.count( key ) == 0 ) || ( descr.size() > 0 ) )
    {
      m_descr_store[key] = descr;
    }
  }
}


// ------------------------------------------------------------------
void
config_block
::set_location( config_block_key_t const& key, std::shared_ptr< std::string > file, int line )
{
  m_def_store[key] = source_location( file, line );
}


// ------------------------------------------------------------------
bool
config_block
::get_location( config_block_key_t const& key,
                std::string& f,
                int& l) const
{
  location_t::const_iterator i = m_def_store.find( key );
  if ( i != m_def_store.end() )
  {
    f = i->second.file();
    l = i->second.line();
    return true;
  }

  return false;
}


// ------------------------------------------------------------------
// Type-specific casting handling, bool specialization
// cast value to bool
template < >
bool
config_block_get_value_cast( config_block_value_t const& value )
{
  static config_block_value_t const true_string = config_block_value_t( "true" );
  static config_block_value_t const false_string = config_block_value_t( "false" );
  static config_block_value_t const yes_string = config_block_value_t( "yes" );
  static config_block_value_t const no_string = config_block_value_t( "no" );
  static config_block_value_t const one_string = config_block_value_t( "1" );
  static config_block_value_t const zero_string = config_block_value_t( "0" );

  // Could also support "on" "off",
  // "oui" "non",
  // "ja" "nein",
  // "si" "no",
  // "sim" "não",
  // "da" "net" (Да нет)

  config_block_value_t value_lower = value;
  std::transform( value_lower.begin(), value_lower.end(), value_lower.begin(), ::tolower );

  if ( ( value_lower == true_string )
       || ( value_lower == yes_string )
       || ( value_lower == one_string ) )
  {
    return true;
  }
  else if ( ( value_lower == false_string )
            || ( value_lower == no_string )
            || ( value_lower == zero_string ) )
  {
    return false;
  }

  throw bad_config_block_cast( "failed to convert from string representation \""
                                + value + "\" to boolean" );
}


// ------------------------------------------------------------------
//   Type specific get_value for string
template < >
std::string
config_block_get_value_cast( config_block_value_t const& value )
{
  // We know config_block_value_t is a string
  return value;
}


// ------------------------------------------------------------------
// private helper method for determining key path prefixes
/**
 * \param key   The key string to check.
 * \param name  The prefix string to check for. Should not include a trailing
 *              block separator.
 * \returns True if the given key does not begin with the given name and is
 *          not a global variable.
 */
bool
does_not_begin_with( config_block_key_t const& key, config_block_key_t const& name )
{
  static config_block_key_t const global_start = config_block::global_value + config_block::block_sep;

  return ! starts_with( key, name + config_block::block_sep ) &&
         ! starts_with( key, global_start );
}


// ------------------------------------------------------------------
// private helper method to strip a block name from a key path
/**
 * Conditionally strip the given subblock name from the given key path. If the
 * given key doesn't start with the given subblock, the given key is returned
 * as is.
 *
 * \param subblock  The subblock string to strip if present.
 * \param key       The key to conditionally strip from.
 * \returns The stripped key name.
 */
config_block_key_t
strip_block_name( config_block_key_t const& subblock, config_block_key_t const& key )
{
  if ( ! starts_with( key, subblock + config_block::block_sep ) )
  {
    return key;
  }

  return key.substr( subblock.size() + config_block::block_sep.size() );
}


// ------------------------------------------------------------------
/// Format config block in a printable stream
void
config_block::
  print( std::ostream& str )
{
  kwiver::vital::config_block_keys_t all_keys = this->available_values();

  VITAL_FOREACH( kwiver::vital::config_block_key_t key, all_keys )
  {
    std::string ro;

    auto const val = this->get_value< kwiver::vital::config_block_value_t > ( key );

    if ( this->is_read_only( key ) )
    {
      ro = "[RO]";
    }

    str << key << ro << " = " << val;

    // Add location information if available
    std::string file;
    int line(0);
    if ( get_location( key, file, line ) )
    {
      str << "  (" << file << ":" << line << ")";
    }

    str << std::endl;
  }
}


} }   // end namespace
