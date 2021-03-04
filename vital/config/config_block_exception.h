// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief config_block related exceptions interface
 */

#ifndef KWIVER_CONFIG_EXCEPTIONS_CONFIG_H
#define KWIVER_CONFIG_EXCEPTIONS_CONFIG_H

#include <vital/config/vital_config_export.h>
#include <vital/vital_config.h>
#include <vital/exceptions/base.h>

#include "config_block_types.h"

#include <string>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/**
 * \brief The base class for all exceptions thrown from \ref kwiver::vital::config_block
 * \ingroup exceptions
 */
class VITAL_CONFIG_EXPORT config_block_exception
  : public vital_exception
{
public:
  /// Constructor.
  config_block_exception() noexcept;
  /// Destructor.
  virtual ~config_block_exception() noexcept;
};

// ------------------------------------------------------------------
/**
 * \brief The inner exception thrown when casting fails.
 * \ingroup exceptions
 */
class VITAL_CONFIG_EXPORT bad_config_block_cast
  : public config_block_exception
{
public:
  /**
   * \brief Constructor.
   * \param reason The reason for the bad cast.
   */
  bad_config_block_cast( std::string const& reason ) noexcept;
  /// Destructor.
  virtual ~bad_config_block_cast() noexcept;
};

// ------------------------------------------------------------------
/**
 * \brief Thrown when a value cannot be converted to the requested type.
 * \ingroup exceptions
 */
class VITAL_CONFIG_EXPORT bad_config_block_cast_exception
  : public config_block_exception
{
public:
  /**
   * \brief Constructor.
   *
   * \param key The key that was requested.
   * \param value The value that was failed to cast.
   * \param type The type that was requested.
   * \param reason The reason for the bad cast.
   */
  bad_config_block_cast_exception( config_block_key_t const&    key,
                                   config_block_value_t const&  value,
                                   std::string const&           type,
                                   std::string const&           reason ) noexcept;
  /// Destructor.
  virtual ~bad_config_block_cast_exception() noexcept;

  /// The requested key name.
  config_block_key_t const m_key;
  /// The value of the requested key.
  config_block_value_t const m_value;
  /// The type requested for the cast.
  std::string const m_type;
  /// The reason for the failed cast.
  std::string const m_reason;
};

// ------------------------------------------------------------------
/**
 * \brief Thrown when a value is requested for a value which does not exist.
 * \ingroup exceptions
 */
class VITAL_CONFIG_EXPORT no_such_configuration_value_exception
  : public config_block_exception
{
public:
  /**
   * \brief Constructor.
   * \param key The key that was requested from the configuration.
   */
  no_such_configuration_value_exception( config_block_key_t const& key ) noexcept;
  /// Destructor.
  virtual ~no_such_configuration_value_exception() noexcept;

  /// The requested key name.
  config_block_key_t const m_key;
};

// ------------------------------------------------------------------
/**
 * \brief Thrown when a value is set but is marked as read-only.
 * \ingroup exceptions
 */
class VITAL_CONFIG_EXPORT set_on_read_only_value_exception
  : public config_block_exception
{
public:
  /**
   * \brief Constructor.
   *
   * \param key The key that was requested from the configuration.
   * \param value The current read-only value of \p key.
   * \param new_value The value that was attempted to be set.
   */
  set_on_read_only_value_exception( config_block_key_t const&   key,
                                    config_block_value_t const& value,
                                    config_block_value_t const& new_value ) noexcept;
  /**
   * \brief Destructor.
   */
  virtual ~set_on_read_only_value_exception() noexcept;

  /// The requested key name.
  config_block_key_t const m_key;
  /// The existing value.
  config_block_value_t const m_value;
  /// The new value.
  config_block_value_t const m_new_value;
};

// ------------------------------------------------------------------
/**
 * \brief Thrown when a value is unset but is marked as read-only.
 * \ingroup exceptions
 */
class VITAL_CONFIG_EXPORT unset_on_read_only_value_exception
  : public config_block_exception
{
public:
  /**
   * \brief Constructor.
   *
   * \param key The key that was requested from the configuration.
   * \param value The current value for \p key.
   */
  unset_on_read_only_value_exception( config_block_key_t const&   key,
                                      config_block_value_t const& value ) noexcept;
  /**
   * \brief Destructor.
   */
  virtual ~unset_on_read_only_value_exception() noexcept;

  /// The requested key name.
  config_block_key_t const m_key;
  /// The existing value.
  config_block_value_t const m_value;
};

// ------------------------------------------------------------------
/**
 * \brief The inner exception thrown when casting fails.
 *
 * \ingroup exceptions
 */
class VITAL_CONFIG_EXPORT bad_configuration_cast
  : public config_block_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param reason The reason for the bad cast.
     */
  bad_configuration_cast(std::string const& reason) noexcept;
    /**
     * \brief Destructor.
     */
    ~bad_configuration_cast() noexcept;
};

// ------------------------------------------------------------------
/// Thrown when a value cannot be converted to the requested type.
/**
 *
 * \ingroup exceptions
 */
class VITAL_CONFIG_EXPORT bad_configuration_cast_exception
  : public config_block_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param key The key that was requested.
     * \param value The value that was failed to cast.
     * \param type The type that was requested.
     * \param reason The reason for the bad cast.
     */
    bad_configuration_cast_exception(kwiver::vital::config_block_key_t const& key,
                                     kwiver::vital::config_block_value_t const& value,
                                     char const* type,
                                     char const* reason) noexcept;
    /**
     * \brief Destructor.
     */
    ~bad_configuration_cast_exception() noexcept;

    /// The requested key name.
    kwiver::vital::config_block_key_t const m_key;
    /// The value of the requested key.
    kwiver::vital::config_block_value_t const m_value;
    /// The type requested for the cast.
    std::string const m_type;
    /// The reason for the failed cast.
    std::string const m_reason;
};

// ------------------------------------------------------------------
/// Base config_io exception class
class VITAL_CONFIG_EXPORT config_block_io_exception
  : public config_block_exception
{
public:
  ///Constructor
  /**
   * \param file_path The path to the file related to this error.
   * \param reason    Reason for the exception.
   */
  config_block_io_exception( config_path_t const& file_path,
                             std::string const&   reason ) noexcept;
  /// Deconstructor
  virtual ~config_block_io_exception() noexcept;

  /// Path to file this exception revolves around.
  config_path_t m_file_path;
  /// Reason for exception
  std::string m_reason;
};

// ------------------------------------------------------------------
/// Exception for when a file could not be found
class VITAL_CONFIG_EXPORT config_file_not_found_exception
  : public config_block_io_exception
{
public:
  /// Constructor
  /**
   * \param file_path The file path that was looked for.
   * \param reason    The reason the file wasn't found.
   */
  config_file_not_found_exception( config_path_t const&  file_path,
                                   std::string const&    reason ) noexcept;
  /// Deconstructor
  virtual ~config_file_not_found_exception() noexcept;
};

// ------------------------------------------------------------------
/// Exception for when a file could not be read for whatever reason.
class VITAL_CONFIG_EXPORT config_file_not_read_exception
  : public config_block_io_exception
{
public:
  ///Constructor
  /**
   * \param file_path The file path on which the read was attempted.
   * \param reason    The reason for the read exception.
   */
  config_file_not_read_exception( config_path_t const& file_path,
                                  std::string const&   reason ) noexcept;
  /// Deconstructor
  virtual ~config_file_not_read_exception() noexcept;
};

// ------------------------------------------------------------------
/// Exception for when a file could not be parsed after being read in
class VITAL_CONFIG_EXPORT config_file_not_parsed_exception
  : public config_block_io_exception
{
public:
  /// Constructor
  /**
   * \param file_path The file path to which the parsing exception occurred.
   * \param reason    The reason for the parsing exception.
   */
  config_file_not_parsed_exception( config_path_t const& file_path,
                                    std::string const&   reason ) noexcept;
  /// Deconstructor
  virtual ~config_file_not_parsed_exception() noexcept;
};

// ------------------------------------------------------------------
/// Exception for when a file was not able to be written
class VITAL_CONFIG_EXPORT config_file_write_exception
  : public config_block_io_exception
{
public:
  /// Constructor
  /**
   * \param file_path The file path to which the write was attempted.
   * \param reason    The reason for the write exception
   */
  config_file_write_exception( config_path_t const&  file_path,
                               std::string const&    reason ) noexcept;
  /// Deconstructor
  virtual ~config_file_write_exception() noexcept;
};

} } // end namespace

#endif
