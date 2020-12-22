// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief config_block related exceptions implementation
 */

#include "config_block.h"

#include <sstream>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
config_block_exception
::config_block_exception() noexcept
{
}

config_block_exception
::~config_block_exception() noexcept
{
}

// ------------------------------------------------------------------
bad_config_block_cast
::bad_config_block_cast( std::string const& reason ) noexcept
  : config_block_exception()
{
  this->m_what = reason;
}

bad_config_block_cast
::~bad_config_block_cast() noexcept
{
}

// ------------------------------------------------------------------
bad_config_block_cast_exception
::bad_config_block_cast_exception( config_block_key_t const&    key,
                                   config_block_value_t const&  value,
                                   std::string const&           type,
                                   std::string const&           reason ) noexcept
: config_block_exception(),
  m_key( key ),
  m_value( value ),
  m_type( type ),
  m_reason( reason )
{
  std::ostringstream sstr;

  sstr << "Failed to cast key \'" << m_key << "\' with value \'"
       << m_value << "\' as a \'" << m_type << "\': " << m_reason << ".";
  m_what = sstr.str();
}

bad_config_block_cast_exception
::~bad_config_block_cast_exception() noexcept
{
}

// ------------------------------------------------------------------
no_such_configuration_value_exception
::no_such_configuration_value_exception( config_block_key_t const& key ) noexcept
  : config_block_exception(),
  m_key( key )
{
  std::ostringstream sstr;

  sstr << "There is no configuration value for the key "
          "\'" << m_key << "\'.";
  m_what = sstr.str();
}

no_such_configuration_value_exception
::~no_such_configuration_value_exception() noexcept
{
}

// ------------------------------------------------------------------
set_on_read_only_value_exception
::set_on_read_only_value_exception( config_block_key_t const&   key,
                                      config_block_value_t const& value,
                                      config_block_value_t const& new_value ) noexcept
  : config_block_exception(),
  m_key( key ),
  m_value( value ),
  m_new_value( new_value )
{
  std::ostringstream sstr;

  sstr << "The key \'" << m_key << "\' "
                                   "was marked as read-only with the value "
                                   "\'" << m_value << "\' was attempted to be "
                                                      "set to \'" << m_new_value << "\'.";
  m_what = sstr.str();
}

set_on_read_only_value_exception
::~set_on_read_only_value_exception() noexcept
{
}

// ------------------------------------------------------------------
unset_on_read_only_value_exception
::unset_on_read_only_value_exception( config_block_key_t const&   key,
                                        config_block_value_t const& value ) noexcept
  : config_block_exception(),
  m_key( key ),
  m_value( value )
{
  std::ostringstream sstr;

  sstr << "The key \'" << m_key << "\' "
                                   "was marked as read-only with the value "
                                   "\'" << m_value << "\' was attempted to be "
                                                      "unset.";
  m_what = sstr.str();
}

unset_on_read_only_value_exception
::~unset_on_read_only_value_exception() noexcept
{
}

// ------------------------------------------------------------------
config_block_io_exception
::config_block_io_exception( config_path_t const& file_path,
                             std::string const& reason ) noexcept
  : config_block_exception(),
  m_file_path( file_path ),
  m_reason( reason )
{
}

config_block_io_exception
::~config_block_io_exception() noexcept
{
}

// ------------------------------------------------------------------
bad_configuration_cast
::bad_configuration_cast(std::string const& reason) noexcept
  : config_block_exception()
{
  m_what = reason;
}

bad_configuration_cast
::~bad_configuration_cast() noexcept
{
}

// ------------------------------------------------------------------
bad_configuration_cast_exception
::bad_configuration_cast_exception(kwiver::vital::config_block_key_t const& key,
                                   kwiver::vital::config_block_value_t const& value,
                                   char const* type,
                                   char const* reason) noexcept
  : config_block_exception()
  , m_key(key)
  , m_value(value)
  , m_type(type)
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "Failed to cast key \'" << m_key << "\' "
          "with value \'" << m_value << "\' as "
          "a \'" << m_type << "\': " << m_reason << ".";

  m_what = sstr.str();
}

bad_configuration_cast_exception
::~bad_configuration_cast_exception() noexcept
{
}

// ------------------------------------------------------------------
config_file_not_found_exception
::config_file_not_found_exception( config_path_t const& file_path, std::string const& reason ) noexcept
  : config_block_io_exception( file_path, reason )
{
  std::ostringstream sstr;

  sstr  << "Could not find file \'" << m_file_path << "\': "
        << m_reason;
  m_what = sstr.str();
}

config_file_not_found_exception
::~config_file_not_found_exception() noexcept
{
}

// ------------------------------------------------------------------
config_file_not_read_exception
::config_file_not_read_exception( config_path_t const& file_path, std::string const& reason ) noexcept
  : config_block_io_exception( file_path, reason )
{
  std::ostringstream sstr;

  sstr  << "Failed to read from file \'" << m_file_path << "\': "
        << m_reason;
  m_what = sstr.str();
}

config_file_not_read_exception
::~config_file_not_read_exception() noexcept
{
}

// ------------------------------------------------------------------
config_file_not_parsed_exception
::config_file_not_parsed_exception( config_path_t const& file_path, std::string const& reason ) noexcept
  : config_block_io_exception( file_path, reason )
{
  std::ostringstream sstr;

  sstr  << "Failed to parse file \'" << m_file_path << "\': "
        << m_reason;
  m_what = sstr.str();
}

config_file_not_parsed_exception
::~config_file_not_parsed_exception() noexcept
{
}

// ------------------------------------------------------------------
config_file_write_exception
::config_file_write_exception( config_path_t const& file_path, std::string const& reason ) noexcept
  : config_block_io_exception( file_path, reason )
{
  std::ostringstream sstr;

  sstr  << "Failed to write to file \'" << m_file_path << "\': "
        << m_reason;
  m_what = sstr.str();
}

config_file_write_exception
::~config_file_write_exception() noexcept
{
}

} } // end namespace
