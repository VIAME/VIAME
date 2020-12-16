/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
