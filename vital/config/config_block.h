/*ckwg +29
 * Copyright 2011-2015 by Kitware, Inc.
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
 * \brief Header for \link kwiver::config_block configuration \endlink object
 */

#ifndef KWIVER_CONFIG_BLOCK_H_
#define KWIVER_CONFIG_BLOCK_H_

#include <vital/config/vital_config_export.h>

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>
#include <ostream>

#include <boost/optional/optional.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

#include "config_block_types.h"
#include "config_block_exception.h"


namespace kwiver {
namespace vital {

class config_block;

/// Shared pointer for the \c config_block class
typedef boost::shared_ptr< config_block > config_block_sptr;

// ----------------------------------------------------------------
/**
 * \brief Configuration value storage structure.
 *
 * A config block represents a hierarchical key/value space, with
 * description and some attributes.
 *
 * Algorithms and other entities use config blocks to specify critical
 * and configurable values. The entries can be created by the program
 * or read from a file. Values in these entries may be subsequently
 * modified (unless set to read-only).
 *
 * The associated shared pointer for this object is \c config_block_sptr
 *
 * When creating a new config block call the static method
 * empty_config() to get a managed config block.
 *
 * A config block contains a logical block or configuration
 * information.  The block may contain sub-blocks of configuration to
 * create a hierarchical configuration space. The levels of the
 * hierarchy are represented in the key name. For example "A:B:C"
 * represents a three layered nested block.
 *
 * Sub-blocks are created from all the entries that match the key
 * prefix.  For example given a block structure A:B:C:D", the
 * sub-block "A:B" would only contain the entries "C:D" that were
 * prefixed with "A:B".
 *
 * The block name is user defined unless the config_block is a
 * sub-block. In that case, the name (get_name()) contains the prefix
 * portion of the key.
 */

class VITAL_CONFIG_EXPORT config_block
  : public boost::enable_shared_from_this< config_block >,
    private boost::noncopyable
{
public:
  /// Create an empty configuration.
  /**
   * This method serves as the CTOR for this class and ensures that
   * there is a shared pointer managing each config block.
   *
   * \param name The name of the configuration block.
   * \returns An empty configuration block.
   */
  static config_block_sptr empty_config( config_block_key_t const& name = config_block_key_t() );

  /// Destructor
  virtual ~config_block();

  /// Get the name of this \c config_block instance.
  config_block_key_t get_name();


  /// Get a subblock from the configuration.
  /**
   * Retrieve an unlinked configuration subblock from the current
   * configuration. Changes made to it do not affect \c *this.
   *
   * \param key The name of the sub-configuration to retrieve.
   * \returns A subblock with copies of the values.
   */
  config_block_sptr subblock( config_block_key_t const& key ) const;


  /// Get a subblock view into the configuration.
  /**
   * Retrieve a view into the current configuration. Changes made to \c *this
   * \b are seen through the view and vice versa.
   *
   * \param key The name of the sub-configuration to retrieve.
   * \returns A subblock which links to the \c *this.
   */
  config_block_sptr subblock_view( config_block_key_t const& key );


  /// Internally cast the value.
  /**
   * Get value from config entry converted to the desired type.
   *
   * \throws no_such_configuration_value_exception Thrown if the requested index does not exist.
   * \throws bad_configuration_cast_exception Thrown if the cast fails.
   *
   * \param key The index of the configuration value to retrieve.
   * \tparam T Desired type for config value.
   * \returns The value stored within the configuration.
   */
  template < typename T >
  T get_value( config_block_key_t const& key ) const;


  /// Cast the value, returning a default value in case of an error.
  /**
   * Get value from config entry converted to the desired type. If
   * the config entry is not there or the value can not be
   * converted, the specified default value is
   * returned. Unfortunately, there is no way to tell what went
   * wrong.
   *
   * \param key The index of the configuration value to retrieve.
   * \param def The value \p key does not exist or the cast fails.
   * \tparam T Desired type for config value.
   * \returns The value stored within the configuration, or \p def if something goes wrong.
   */
  template < typename T >
  T get_value( config_block_key_t const& key, T const& def ) const VITAL_NOTHROW;


  /// Get the description associated to a value
  /**
   * If the provided key has no description associated with it, an empty
   * \c config_block_description_t value is returned.
   *
   * \throws no_such_configuration_value_exception Thrown if the requested
   *                                               key does not exist.
   *
   * \param key The name of the parameter to get the description of.
   * \returns The description of the requested key.
   */
  config_block_description_t get_description( config_block_key_t const& key ) const;


  /// Set a value within the configuration.
  /**
   * If this key already exists, has a description and no new description
   * was passed with this \c set_value call, the previous description is
   * retained. We assume that the previous description is still valid and
   * this a value overwrite. If it is intended for the description to also
   * be overwritted, an \c unset_value call should be performed on the key
   * first, and then this \c set_value call.
   *
   * \throws set_on_read_only_value_exception Thrown if \p key is marked as read-only.
   *
   * \postconds
   * \postcond{<code>this->get_value<value_t>(key) == value</code>}
   * \endpostconds
   *
   * \param key The index of the configuration value to set.
   * \param value The value to set for the \p key.
   * \param descr Description of the key. If this is set, we will override
   *              any existing description for the given key. If a
   *              description for the given key already exists and nothing
   *              was provided for this parameter, the existing description
   *              is maintained.
   */
  template < typename T >
  void set_value( config_block_key_t const& key,
                  T const& value,
                  config_block_description_t const& descr = config_block_key_t() );

  /// Remove a value from the configuration.
  /**
   * \throws unset_on_read_only_value_exception Thrown if \p key is marked as read-only.
   * \throws no_such_configuration_value_exception Thrown if the requested index does not exist.
   *
   * \postconds
   * \postcond{<code>this->get_value<T>(key)</code> throws \c no_such_configuration_value_exception}
   * \endpostconds
   *
   * \param key The index of the configuration value to unset.
   */
  void unset_value( config_block_key_t const& key );

  /// Query if a value is read-only.
  /**
   *
   * \param key The key of the value query.
   * \returns True if \p key is read-only, false otherwise.
   */
  bool is_read_only( config_block_key_t const& key ) const;

  /// Set the value within the configuration as read-only.
  /**
   * This method sets the specified configuration key as read
   * only. This prevents the value from being changed at a later
   * time by data from a config file or programatically changing the
   * value.
   *
   * \postconds
   * \postcond{<code>this->is_read_only(key) == true</code>}
   * \endpostconds
   *
   * \param key The key of the value to mark as read-only.
   */
  void mark_read_only( config_block_key_t const& key );

  /// Merge the values in \p config into the current config.
  /**
   * This method merges the values from the specified config block
   * into this config block. Both the values and descriptions from
   * the specified block are merged.
   *
   * Values for keys that do not exist in this block are
   * created. Values for keys that already exist in this block are
   * overwritten.  If the entry in this config is marked as
   * read-only, an exception is thrown and the merge operation is
   * left as partially complete. If an entry in the specified config
   * block is marked as read-only, that attribute is not copied to
   * this block.
   *
   * \note Any values currently set within \c *this will be overwritten if conflicts occur.
   *
   * \throws set_on_read_only_value_exception Thrown if \p key is marked as read-only.
   *
   * \postconds
   * \postcond{\c this->available_values() ⊆ \c config->available_values()}
   * \endpostconds
   *
   * \param config The other configuration.
   */
  void merge_config( config_block_sptr const& config );

  ///Return the values available in the configuration.
  /**
   * This method returns a list of all config entry keys available
   * in this config block. The returned list contains a copy of the
   * keys and is yours to use for any purpose.
   *
   * \returns All of the keys available within the block.
   */
  config_block_keys_t available_values() const;

  /// Check if a value exists for \p key.
  /**
   * \param key The index of the configuration value to check.
   * \returns Whether the key exists.
   */
  bool has_value( config_block_key_t const& key ) const;

  /// The separator between blocks.
  static config_block_key_t const block_sep;

  /// The magic group for global parameters.
  static config_block_key_t const global_value;

  /// Format config in printable form
  /**
   * This method formats the config entries onto the supplied stream.
   *
   * \paaram str Stream to accept formated text.
   */
  void print( std::ostream & str );

private:
  /// Internal constructor
  VITAL_CONFIG_NO_EXPORT config_block( config_block_key_t const& name, config_block_sptr parent );

  /// Private helper method to extract a value for a key
  /**
   * \param key key to find the associated value to.
   * \returns boost::none if the key doesn't exist or the key's value.
   */
  boost::optional< config_block_value_t > find_value( config_block_key_t const& key ) const;
  /// private value getter function
  /**
   * \param key key to get the associated value to.
   * \returns key's value or an empty config_block_value_t if the key is not found.
   */
  VITAL_CONFIG_NO_EXPORT config_block_value_t i_get_value( config_block_key_t const& key ) const;
  /// private key/value setter
  /**
   * \param key key to set a value to
   * \param value the value as a config_block_value_t
   * \param descr optional description of the key.
   */
  void i_set_value( config_block_key_t const& key,
                    config_block_value_t const& value,
                    config_block_description_t const& descr = config_block_key_t() );

  typedef std::map< config_block_key_t, config_block_value_t > store_t;
  typedef std::set< config_block_key_t > ro_list_t;

  // Used to manage views of config blocks. If a parent is specified,
  // then this is a view of that config block.
  config_block_sptr m_parent;

  // Name of this config block. This is used when is sub-block is
  // created to hold the higher levels of the config key.
  //
  // Example: given config block a:b:c:d
  // request sub-block a:b, m_name becomes "a:b"
  config_block_key_t m_name;

  // key => string value map
  store_t m_store;

  // key => to description map
  store_t m_descr_store;

  // list of keys that are read-only
  ro_list_t m_ro_list;
};


// ------------------------------------------------------------------
/// Default cast handling of configuration values.
/**
 * The specified value is converted into a form suitable for this
 * config block, i.e.a string. boost::lexical_cast does the work of
 * converting the value. If the type is something other than a simple
 * type, then an input and output operator must be available for that
 * type.
 *
 * \note Do not use this in user code. Use \ref config_block_cast instead.
 * \param value The value to convert.
 * \tparam R Type returned.
 * \tparam T Parameter type.
 * \returns The value of \p value in the requested type.
 */
template < typename R, typename T >
inline
R
config_block_cast_default( T const& value )
{
  try
  {
    return boost::lexical_cast< R > ( value );
  }
  catch ( boost::bad_lexical_cast const& e )
  {
    throw bad_config_block_cast( e.what() );
  }
}


// ------------------------------------------------------------------
/// Cast a configuration value to the requested type.
/**
 * \throws bad_configuration_cast Thrown when the conversion fails.
 * \param value The value to convert.
 * \tparam R Type returned.
 * \tparam T Parameter type.
 * \returns The value of \p value in the requested type.
 */
template < typename R, typename T >
inline
R
config_block_cast( T const& value )
{
  return config_block_cast_default< R, T > ( value );
}


// ------------------------------------------------------------------
/// Type-specific casting handling, config_block_value_t->bool specialization
/**
 * This is the \c bool to \c config_block_value_t specialization to handle
 * \c true, \c false, \c yes and \c no literal conversion versus just
 * \c 1 and \c 0 (1 and 0 still handled if provided).
 *
 * \note Do not use this in user code. Use \ref config_block_cast instead.
 * \param value The value to convert.
 * \returns The value of \p value in the requested type.
 */
template < >
VITAL_CONFIG_EXPORT
bool config_block_cast( config_block_value_t const& value );


// ------------------------------------------------------------------
/// Type-specific casting handling, bool->cb_value_t specialization
/**
 * This is the \c config_block_value_t to \c bool specialization that outputs
 * \c true and \c false literals instead of 1 or 0.
 *
 * \note Do not use this in user code. Use \ref config_block_cast instead.
 * \param value The value to convert.
 * \returns The value of \p value as either "true" or "false".
 */
template < >
inline
config_block_value_t
config_block_cast( bool const& value )
{
  return value ? "true" : "false";
}


// ------------------------------------------------------------------
// Internally cast the value.
template < typename T >
T
config_block
  ::get_value( config_block_key_t const& key ) const
{
  boost::optional< config_block_value_t > value = find_value( key );

  if ( ! value )
  {
    throw no_such_configuration_value_exception( key );
  }

  try
  {
    return config_block_cast< T, config_block_value_t > ( *value );
  }
  catch ( bad_config_block_cast const& e )
  {
    throw bad_config_block_cast_exception( key, *value, typeid( T ).name(), e.what() );
  }
}


// ------------------------------------------------------------------
// Cast the value, returning a default value in case of an error.
template < typename T >
T
config_block
  ::get_value( config_block_key_t const& key, T const& def ) const VITAL_NOTHROW
{
  try
  {
    return get_value< T > ( key );
  }
  catch ( ... )
  {
    return def;
  }
}


// ------------------------------------------------------------------
// Set a value within the configuration.
template < typename T >
void
config_block
  ::set_value( config_block_key_t const&          key,
               T const&                           value,
               config_block_description_t const&  descr )
{
  this->i_set_value( key, config_block_cast< config_block_value_t, T > ( value ), descr );
}


}
}
#endif // KWIVER_CONFIG_BLOCK_H_
