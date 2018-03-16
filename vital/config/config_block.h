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
 * @file
 * \brief Header for \link kwiver::vital::config_block configuration \endlink object
 */

#ifndef KWIVER_CONFIG_BLOCK_H_
#define KWIVER_CONFIG_BLOCK_H_

#include <vital/config/vital_config_export.h>
#include <vital/noncopyable.h>
#include <vital/util/source_location.h>
#include <vital/util/tokenize.h>

#include "config_block_types.h"
#include "config_block_exception.h"

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>
#include <ostream>
#include <memory>
#include <exception>
#include <sstream>

namespace kwiver {
namespace vital {

template < typename R >
R config_block_get_value_cast_default( config_block_value_t const& value );

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
 *
 * config_block supports simple data types by using the input and
 * output operators for conversions to/from a string
 * representation. More complicated data can be supported by
 * specializing the config_block_set_value_cast() and
 * config_block_get_value_cast() functions.
 *
 * \sa config_block_get_value_cast()
 * \sa config_block_set_value_cast()
 */

class VITAL_CONFIG_EXPORT config_block
  : public std::enable_shared_from_this< config_block >,
    private kwiver::vital::noncopyable
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
  T get_value( config_block_key_t const& key, T const& def ) const noexcept;


  /**
   * \brief Convert string to enum value.
   *
   * \param key The index of the configuration value to retrieve.
   * \tparam C Type of the enum converter. Must be derived from
   * enum_converter struct.
   * \return
   */
  template < typename C>
  typename C::enum_type get_enum_value( const config_block_key_t& key ) const;


  /**
   * \brief Convert string to vector of values.
   *
   * Convert config string into a vector of values of the same type. This method
   * splits the config string associated with the key using the supplied delimeter
   * string. Each of these resulting strings is converted to the templated type and
   * added to the output vector. The final set of values is returned in the vector.
   *
   * \param key The index of the configuration value to retrieve.
   * \param delim List of delimeter characters for splitting vector elements.
   * \tparam T Type of vector element.
   */
  template< typename T >
  std::vector< T > get_value_as_vector( config_block_key_t const& key, const std::string& delim = " " ) const;


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
   * The specified value is set for the specified key.
   *
   * If this key already exists, has a description and no new description
   * was passed with this \c set_value call, the previous description is
   * retained (we assume that the previous description is still valid and
   * this is just a value overwrite).
   *
   * \throws set_on_read_only_value_exception Thrown if \p key is marked as
   *   read-only.
   *
   * \postconds
   * \postcond{<code>this->get_value<value_t>(key) == value</code>}
   * \endpostconds
   *
   * \param key The index of the configuration value to set.
   * \param value The value to set for the \p key.
   * \param descr Description of the key. If this is set, we will override any
   *   existing description for the given key. If a description for the given
   *   key already exists and nothing was provided for this parameter, the
   *   existing description is maintained.
   */
  template < typename T >
  void set_value( config_block_key_t const&         key,
                  T const&                          value,
                  config_block_description_t const& descr );

  /// Set a value within the configuration.
  /**
   * The specified value is set for the specified key.
   *
   * If this key already exists and has a description, the previous description
   * is retained (we assume that the previous description is still valid and
   * this a value overwrite). If it is intended for the description to also
   * be overwritten, the other \c set_value function that has a parameter for
   * description setting should be used.
   *
   * \throws set_on_read_only_value_exception Thrown if \p key is marked as read-only.
   *
   * \postconds
   * \postcond{<code>this->get_value<value_t>(key) == value</code>}
   * \endpostconds
   *
   * \param key The index of the configuration value to set.
   * \param value The value to set for the \p key.
   */
  template < typename T >
  void set_value( config_block_key_t const& key,
                  T const&                  value);

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


  /// Get difference between this and other config block.
  /**
   * This method determines the difference between two config blocks
   * (this - other) and returns a new config block that contains all
   * entries that are in \b this config block but not in the other.
   *
   * \param other The config block to be differenced with.
   */
  config_block_sptr difference_config( const config_block_sptr other ) const;


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
   * The existence of a key is checked in the config block.
   *
   * \param key The key to check for existence.
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
   * \param str Stream to accept formated text.
   */
  void print( std::ostream & str );


  /// Set source file location where entry is defined.
  /**
   * This method adds the source file location where a config entry
   * originated.
   *
   * \param key Config entry key string
   * \param file Name of defining file
   * \param line Line number in file
   */
  void set_location( config_block_key_t const& key, std::shared_ptr< std::string > file, int line );
  void set_location( config_block_key_t const& key, const kwiver::vital::source_location& loc );


  /// Get file location where config key was defined.
  /**
   * This method returns the location where the specified config entry
   * was defined. If it is no location for the definition of the
   * symbol, the output parameters are unchanged.
   *
   * \param[in] key Name of the config entry
   * \param[out] file Name of the last file where this symbol was defined
   * \param[out] line Line number in file of definition
   *
   * \return \b true if the location is available.
   */
  bool get_location( config_block_key_t const& key,
                     std::string& file,
                     int& line) const;


  /// Get file location where config key was defined.
  /**
   * This method returns the location where the specified config entry
   * was defined. If it is no location for the definition of the
   * symbol, the output parameters are unchanged.
   *
   * \param[in] key Name of the config entry
   * \param[out] loc Location of where this entry was defined.
   *
   * \return \b true if the location is available.
   */
  bool get_location( config_block_key_t const& key, kwiver::vital::source_location& loc ) const;

private:
  /// Internal constructor
  VITAL_CONFIG_NO_EXPORT config_block( config_block_key_t const& name, config_block_sptr parent );

  /// Private helper method to extract a value for a key
  /**
   * \param[in] key key to find the associated value to.
   * \param[out] val value associated with key
   * \returns \b true if key is found and value returned, \b false if key not found.
   */
  bool find_value( config_block_key_t const& key,  config_block_value_t& val ) const;

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

  /**
   * @brief Copies config entry to this config block.
   *
   * This function copies one config entry, as specified by the \b key
   * from the specified config block to this block.
   *
   * @param key Specifies the config entry to copy.
   * @param from The source config block.
   */
  void copy_entry( config_block_key_t const& key,
                   const config_block_sptr from );

  void copy_entry( config_block_key_t const& key,
                   const config_block* from );

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


  typedef std::map< config_block_key_t, kwiver::vital::source_location > location_t;

  // location where key was defined.
  location_t m_def_store;
};

// ==================================================================
// ---- get value group ----
// ------------------------------------------------------------------
/** \defgroup get_value_group Get Config Value Group
 * Functions to get typed values from a config entry.
 * @{
 */

/// Default cast handling for getting configuration values.
/**
 * The specified value is converted into a form suitable for this
 * config block. If the type is something other than a simple type,
 * then an input and output operator must be available for that type.
 *
 * Relying on input and output operators can be unfortunate if they
 * have already been implemented with a more wordy format. To get
 * around this problem define a specialized version of
 * config_block_get_value_cast<>() for your specific type.
 *
 * \note Do not use this in user code. Use config_block_get_value_cast() instead.
 *
 * \param value The value to convert.
 * \tparam R Type returned.
 * \returns The value of \p value in the requested type.
 */
template < typename R >
inline
R
config_block_get_value_cast_default( config_block_value_t const& value )
{
  try
  {
    std::stringstream interpreter;
    interpreter << value; // string value

    R result;
    interpreter >> result;
    if( interpreter.fail() )
    {
      throw bad_config_block_cast( "failed to convert from string representation \"" + value + "\"" );
    }

    return result;
  }
  catch( std::exception& e )
  {
    throw bad_config_block_cast( e.what() );
  }
}


// ------------------------------------------------------------------
/// Cast a configuration value to the requested type.
/**
 * This method converts the config block value from its native string
 * representation to the desired type.
 *
 * If the default implementation (using input operator) does not work
 * for your data type, then write a specialized version of this
 * function to do the conversion as show in the following example.
 *
 * Example:
\code
template<>
timestamp
config_block_get_value_cast( config_block_value_t const& value )
{
  std::stringstream str;
  str << value;

  kwiver::vital::time_us_t t;
  str >> t;
  obj.set_time( t );

  kwiver::vital::frame_id_t f;
  str >> f;
  obj.set_frame( f );

  return str;
}
\endcode
 *
 * \throws bad_configuration_cast Thrown when the conversion fails.
 * \param value The value to convert.
 * \tparam R Type returned.
 * \returns The value of \p value in the requested type.
 */
template < typename R >
inline
R
config_block_get_value_cast( config_block_value_t const& value )
{
  return config_block_get_value_cast_default< R > ( value );
}


// ------------------------------------------------------------------
/// Type-specific casting handling for config_block_value_t->bool specialization
/**
 * This is the \c bool to \c config_block_value_t specialization to handle
 * \c true, \c false, \c yes and \c no literal conversion versus just
 * \c 1 and \c 0 (1 and 0 still handled if provided).
 *
 * \param value The value to convert.
 * \returns The value of \p value in the requested type.
 */
template < >
VITAL_CONFIG_EXPORT
bool config_block_get_value_cast( config_block_value_t const& value );


// ------------------------------------------------------------------
/// Type-specific cast handling for config_block_value_t->string specialization
/**
 * This function converts from a string to a string.
 *
 * @param value String to be converted
 *
 * @return Resulting string
 */
template < >
VITAL_CONFIG_EXPORT
std::string config_block_get_value_cast( config_block_value_t const& value );


// ------------------------------------------------------------------
// Internally cast the value.
template < typename T >
T
config_block
::get_value( config_block_key_t const& key ) const
{
  config_block_value_t value;
  if ( ! find_value(key, value ) )
  {
    throw no_such_configuration_value_exception( key );
  }

  try
  {
    // Convert config block value to requested type
    return config_block_get_value_cast< T > ( value );
  }
  catch ( bad_config_block_cast const& e )
  {
    // Upgrade exception by adding more known details.
    throw bad_config_block_cast_exception( key, value, typeid( T ).name(), e.what() );
  }
}


// ------------------------------------------------------------------
template < typename C >
typename C::enum_type
config_block
::get_enum_value( const config_block_key_t& key ) const
{
  return C().from_string( get_value < std::string >( key ) );
}


// ------------------------------------------------------------------
template< typename T >
std::vector< T >
config_block
::get_value_as_vector( config_block_key_t const& key, const std::string& delim ) const
{
  config_block_value_t val = get_value< std::string >( key );

  std::vector< std::string> sv;
  // Split string by delimeter into vector of strings
  tokenize( val, sv, delim, kwiver::vital::TokenizeTrimEmpty );

  std::vector< T > val_vector;
  // iterate over all strings and convert to target type
  for (std::string str : sv )
  {
    T val = config_block_get_value_cast<T>( str );
    val_vector.push_back( val );
  }

  return val_vector;
}


// ------------------------------------------------------------------
// Cast the value, returning a default value in case of an error.
template < typename T >
T
config_block
::get_value( config_block_key_t const& key, T const& def ) const noexcept
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
//@}

// ==================================================================
//  ---- set value group ----
// ------------------------------------------------------------------
  /** \defgroup set_value_group Set Config Value Group
 * Functions to set typed values in a config entry.
 * @{
 */

/// Default cast handling for setting config values
/**
 * The supplied value is converted from its native type to a string
 * using the output operator.
 *
 * Relying on input and output operators can be unfortunate if they
 * have already been implemented with a more wordy format. To get
 * around this problem define a specialized version of
 * config_block_set_value_cast<>() for your specific type.
 *
 * \note Do not use this in user code. Use
 * config_block_set_value_cast() instead.
 *
 * \param value   Value to be converted to string representation.
 * \tparam T Type to be converted.
 *
 * \return String representation of the input value.
 */
template < typename T >
inline
config_block_value_t
config_block_set_value_cast_default( T const& value )
{
  std::stringstream val_str;

  try
  {
    val_str << value;
    if ( val_str.fail() )
    {
      throw bad_config_block_cast( "failed to convert value to string representation" );
    }

    return val_str.str();
  }
    catch( std::exception& e )
  {
    throw bad_config_block_cast( e.what() );
  }
}


// ------------------------------------------------------------------
/// Cast a configuration value to the requested type.
/**
 * This method converts the user supplied value from its native form
 * to a string representation to the desired type.
 *
 * If the default implementation (using output operator) does not work
 * for your data type, then write a specialized version of this
 * function to do the conversion.
 *
 * Example:
\code
template<>
config_block_value_t
config_block_set_value_cast( timestamp const& value )
{
  std::stringstream str;

  str << value.get_time() << " " << value.get_frame();

  return str.str();
}
\endcode
 *
 * \throws bad_configuration_cast Thrown when the conversion fails.
 * \param value The value to convert.
 * \tparam T Type to be converted.
 * \returns The value of \p value as a string.
 */
template < typename T >
inline
config_block_value_t
config_block_set_value_cast( T const& value )
{
  return config_block_set_value_cast_default< T > ( value );
}


// ------------------------------------------------------------------
// Set a value within the configuration.
template < typename T >
inline
void
config_block
::set_value( config_block_key_t const&          key,
             T const&                           value )
{
  // Need to convert value (type T) to string
  config_block_value_t val_str = config_block_set_value_cast< T > ( value );

  this->i_set_value( key,  val_str, config_block_description_t() ); // we know that the value is a string
}


// ------------------------------------------------------------------
// Set a value within the configuration.
template < typename T >
inline
void
config_block
::set_value( config_block_key_t const&          key,
             T const&                           value,
             config_block_description_t const&  descr )
{
  // Need to convert value (type T) to string
  config_block_value_t val_str = config_block_set_value_cast< T > ( value );

  this->i_set_value( key,  val_str, descr ); // we know that the value is a string
}


// ------------------------------------------------------------------
/// Type-specific handling, bool->config_block_value_t specialization
/**
 * This is the \c config_block_value_t to \c bool specialization that outputs
 * \c true and \c false literals instead of 1 or 0.
 *
 * \param key The configuration key string
 * \param value The value to convert.
 * \param descr Configuration item descrription
 */
template < >
inline
void
config_block
::set_value( config_block_key_t const&          key,
             bool const&                        value,
             config_block_description_t const&  descr )
{
  this->i_set_value( key, (value ? "true" : "false"), descr );
}


// ------------------------------------------------------------------
/// Type-specific handling, string->config_block_value_t specialization
/**
 * This is the \c config_block_value_t to \c string specialization that outputs
 * the value string directly.
 *
 * \param key The configuration key string
 * \param value The value to convert.
 * \param descr Configuration item descrription
 */
template < >
inline
void
config_block
::set_value( config_block_key_t const&          key,
             std::string const&                 value,
             config_block_description_t const&  descr )
{
  this->i_set_value( key, value, descr );
}
//@}

} }

#endif // KWIVER_CONFIG_BLOCK_H_
