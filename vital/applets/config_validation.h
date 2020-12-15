// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOLS_CONFIG_VALIDATION_H
#define KWIVER_TOOLS_CONFIG_VALIDATION_H

#include <vital/applets/kwiver_tools_applet_export.h>

#include <vital/config/config_block.h>
#include <vital/logger/logger.h>

#include <string>

namespace kwiver {
namespace tools {

/// Validate a required input file for a key name in the given config block
/**
 * Verify that the config block contains the key \a name, that the value is
 * not the empty string, and that it refers to a valid file on disk
 *
 * \param name   The key name to look for in the configuration block
 * \param config The configuration block to check for the key
 * \param logger The logger handle to log error messages to
 *
 * \returns true if sucessfully validated
 */
KWIVER_TOOLS_APPLET_EXPORT
bool
validate_required_input_file(std::string const& name,
                             kwiver::vital::config_block const& config,
                             kwiver::vital::logger_handle_t logger);

/// Validate an optional input file for a key name in the given config block
/**
 * If the config block contains the key \a name and the value is not empty
 * then verify that it refers to a valid file on disk
 *
 * \param name   The key name to look for in the configuration block
 * \param config The configuration block to check for the key
 * \param logger The logger handle to log error messages
 *
 * \returns true if file not set or if set and sucessfully validated
 */
KWIVER_TOOLS_APPLET_EXPORT
bool
validate_optional_input_file(std::string const& name,
                             kwiver::vital::config_block const& config,
                             kwiver::vital::logger_handle_t logger);

/// Validate a required output file for a key name in the given config block
/**
 * Verify that the config block contains the key \a name, that the value is
 * not the empty string, and that it refers to a valid path for writing an
 * output file.  If \a make_directory is true then construct directories
 * as needed to make a valid path.  If \a test_write is true then try to
 * open the file for writing to confirm write permissions.
 *
 * \param name           The key name to look for in the configuration block
 * \param config         The configuration block to check for the key
 * \param logger         The logger handle to log error messages
 * \param make_directory If true, create any missing directory structure
 * \param test_write     If true, test create the file to verify write
 *                       permission
 * \returns true if sucessfully validated
 */
KWIVER_TOOLS_APPLET_EXPORT
bool
validate_required_output_file(std::string const& name,
                              kwiver::vital::config_block const& config,
                              kwiver::vital::logger_handle_t logger,
                              bool make_directory = true,
                              bool test_write = true);

/// Validate an optional output file for a key name in the given config block
/**
 * If the config block contains the key \a name and the value is not empty
 * then verify that it refers to a valid path for writing an output file.
 * If \a make_directory is true then construct directories as needed to make
 * a valid path.  If \a test_write is true then try to open the file for
 * writing to confirm write permissions.
 *
 * \param name           The key name to look for in the configuration block
 * \param config         The configuration block to check for the key
 * \param logger         The logger handle to log error messages
 * \param make_directory If true, create any missing directory structure
 * \param test_write     If true, test create the file to verify write
 *                       permission
 * \returns true if file not set or if set and sucessfully validated
 */
KWIVER_TOOLS_APPLET_EXPORT
bool
validate_optional_output_file(std::string const& name,
                              kwiver::vital::config_block const& config,
                              kwiver::vital::logger_handle_t logger,
                              bool make_directory = true,
                              bool test_write = true);

/// Validate a required input directory for a key name in the given config block
/**
 * Verify that the config block contains the key \a name, that the value is
 * not the empty string, and that it refers to a valid directory on disk
 *
 * \param name   The key name to look for in the configuration block
 * \param config The configuration block to check for the key
 * \param logger The logger handle to log error messages to
 *
 * \returns true if sucessfully validated
 */
KWIVER_TOOLS_APPLET_EXPORT
bool
validate_required_input_dir(std::string const& name,
                            kwiver::vital::config_block const& config,
                            kwiver::vital::logger_handle_t logger);

/// Validate an optional input directory for a key name in the given config block
/**
 * If the config block contains the key \a name and the value is not empty
 * then verify that it refers to a valid directory on disk
 *
 * \param name   The key name to look for in the configuration block
 * \param config The configuration block to check for the key
 * \param logger The logger handle to log error messages
 *
 * \returns true if directory not set or if set and sucessfully validated
 */
KWIVER_TOOLS_APPLET_EXPORT
bool
validate_optional_input_dir(std::string const& name,
                            kwiver::vital::config_block const& config,
                            kwiver::vital::logger_handle_t logger);

/// Validate a required output directory for a key name in the given config block
/**
 * Verify that the config block contains the key \a name, that the value is
 * not the empty string, and that it refers to a valid directory for writing
 * output files.  If \a make_directory is true then construct directories
 * as needed to make a valid path.
 *
 * \param name           The key name to look for in the configuration block
 * \param config         The configuration block to check for the key
 * \param logger         The logger handle to log error messages
 * \param make_directory If true, create any missing directory structure
 *
 * \returns true if sucessfully validated
 */
KWIVER_TOOLS_APPLET_EXPORT
bool
validate_required_output_dir(std::string const& name,
                             kwiver::vital::config_block const& config,
                             kwiver::vital::logger_handle_t logger,
                             bool make_directory = true);

/// Validate an optional output directory for a key name in the given config block
/**
 * If the config block contains the key \a name and the value is not empty
 * then verify that it refers to a valid directory for writing output files.
 * If \a make_directory is true then construct directories as needed to make
 * a valid path.
 *
 * \param name           The key name to look for in the configuration block
 * \param config         The configuration block to check for the key
 * \param logger         The logger handle to log error messages
 * \param make_directory If true, create any missing directory structure
 *
 * \returns true if directory not set or if set and sucessfully validated
 */
KWIVER_TOOLS_APPLET_EXPORT
bool
validate_optional_output_dir(std::string const& name,
                             kwiver::vital::config_block const& config,
                             kwiver::vital::logger_handle_t logger,
                             bool make_directory = true);

}
}
#endif
