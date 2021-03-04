// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file interface for config difference class
 */

#ifndef CONFIG_CONFIG_DIFFERENCE_H
#define CONFIG_CONFIG_DIFFERENCE_H

#include <vital/config/vital_config_export.h>
#include <vital/config/config_block.h>
#include <vital/vital_types.h>

namespace kwiver {
namespace vital {

// -----------------------------------------------------------------
/// Config difference class for validating provided config.
/**
 * This class performs differencing between a reference set of config
 * entries, such as your expected config block, and a config block
 * that is passed to the class. These two config blocks are diff'd two
 * ways to get the list of expected config entries that are not
 * supplied and the list of config entries that are supplied but not
 * requested.
 *
 * It is up to the user to determine what these two lists mean and how
 * to report the differences.
 *
 * There are a few cases, such as processes and algorithms, where the
 * expected config is known and established in the code. In these
 * cases it is useful to know if there are any extra config items
 * supplied that are not expected. Must usually these extra items are
 * misspellings of expected ones, or from general confusion about what
 * config entries are really required.
 *
 * The following example shows how the config difference might be used
 * to detect config errors.
 *
 \code
  //                                    ref-config                received-config
  kwiver::vital::config_difference cd( this->get_configuration(), config );
  const auto key_list = cd.extra_keys();
  if ( ! key_list.empty() )
  {
    // This may be considered an error in some cases
    LOG_WARN( logger(), "Additional parameters found in config block that are not required or desired: "
              << kwiver::vital::join( key_list, ", " ) );
  }

  key_list = cd.unspecified_keys();
  if ( ! key_list.empty() )
  {
    LOG_WARN( logger(), "Parameters that were not supplied in the config, using default values: "
              << kwiver::vital::join( key_list, ", " ) );
  }
 \endcode
 *
 */
class VITAL_CONFIG_EXPORT config_difference
{
public:
  config_difference( const config_block_sptr reference, const config_block_sptr other );
  config_difference( config_block_keys_t const& reference, const config_block_sptr other );
  virtual ~config_difference();

  /**
   * @brief Get list of extra config keys.
   *
   * This method returns the list of keys that are in the \b other
   * config block but are not in the \b reference config block.
   *
   * This would be used in validating a config block against a set of
   * expected keys. This list contains the keys that are not expected
   * to be in the \b reference config block. These keys have been
   * provided even though they have not been requested.
   *
   * The entries corresponding to these keys can be found in the \b
   * other config block.
   *
   * The two lists are created when this class is instantiated.
   *
   * @return List of config keys.
   */
  config_block_keys_t extra_keys() const;

  /**
   * @brief Get list of unspecified config keys.
   *
   * This method returns the list of keys that are in the \b reference
   * config block but are not in the \b other config block.
   *
   * This would be used in validating a config block against a set of
   * expected keys. This list contains the key for entries that are
   * expected to be set but are not present.
   *
   * The entries corresponding to these keys can be found in the \b
   * reference config block.
   *
   * @return List of config keys.
   */
  config_block_keys_t unspecified_keys() const;

  /**
   * @brief Issue log warnings for extra config keys.
   *
   * this method will issue a single log warning with a list of config
   * keys that are unexpected.
   *
   * @param logger - A logger handle
   *
   * @return True if warning was generated
   */
  bool warn_extra_keys(kwiver::vital::logger_handle_t logger) const;

  /**
   * @brief Issue log warning for unspecified keys
   *
   * this methos will issue a single log warning with a list of config
   * keys that are unspecified.
   *
   * @param logger - A logger handle
   *
   * @return True if warning was generated
   */
  bool warn_unspecified_keys(kwiver::vital::logger_handle_t logger) const;

private:
  config_block_keys_t m_extra_keys;
  config_block_keys_t m_missing_keys;

}; // end class config_difference

} } // end namespace

#endif // CONFIG_CONFIG_DIFFERENCE_H
