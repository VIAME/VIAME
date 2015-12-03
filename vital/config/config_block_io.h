/*ckwg +29
 * Copyright 2013-2014 by Kitware, Inc.
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
 * \brief IO Operation utilities for \c kwiver::config
 *
 * \todo Describe format here.
 */

#ifndef KWIVER_CONFIG_BLOCK_IO_H_
#define KWIVER_CONFIG_BLOCK_IO_H_

#include <vital/config/vital_config_export.h>
#include "config_block.h"

#include <ostream>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Read in a configuration file, producing a \c config_block object
/**
 *
 *
 * \throws config_file_not_found_exception
 *    Thrown when the file could not be found on the file system.
 * \throws config_file_not_read_exception
 *    Thrown when the file could not be read or parsed for whatever reason.
 *
 * \param file_path   The path to the file to read in.
 * \return A \c config_block object representing the contents of the read-in file.
 */
config_block_sptr VITAL_CONFIG_EXPORT read_config_file(
  config_path_t const& file_path );


// ------------------------------------------------------------------
/// Read in (a) configuration file(s), producing a \c config_block object
/**
 * This function reads one or more configuration files from platform specific
 * standard locations and from locations specified by the \c VITAL_CONFIG_PATH
 * environmental variable. \c VITAL_CONFIG_PATH is searched first, followed by
 * the user-specific location(s), followed by the machine-wide location(s).
 *
 * \throws config_file_not_found_exception
 *    Thrown when the no matching file could be found in the searched paths.
 * \throws config_file_not_read_exception
 *    Thrown when a file could not be read or parsed for whatever reason.
 *
 * \param file_name
 *   The name to the file(s) to read in.
 * \param application_name
 *   The application name, used to build the list of standard locations to be
 *   searched.
 * \param application_version
 *   The application version number, used to build the list of standard
 *   locations to be searched.
 * \param install_prefix
 *   The prefix to which the application is installed (should be one directory
 *   higher than the location of the executing binary).
 * \param merge
 *   If \c true, search all locations for matching config files, merging their
 *   contents, with files earlier in the search order taking precedence. If
 *   \c false, read only the first matching file.
 *
 * \return
 *   A \c config_block object representing the contents of the read-in file.
 */
config_block_sptr
VITAL_CONFIG_EXPORT read_config_file(
  std::string const& file_name,
  std::string const& application_name,
  std::string const& application_version,
  config_path_t const& install_prefix = config_path_t(),
  bool merge = true );


// ------------------------------------------------------------------
/// Output to file the given \c config_block object to the specified file path
/**
 * This function writes the specified config block to the specified
 * file.  If a key has an associated description, it will be written
 * as a comment.  The key and value strings are written in a format
 * that can be read by the read_config_file() function.
 *
 * If a file exists at the target location, it will be overwritten. If the
 * containing directory of the given path does not exist, it will be created
 * before the file is opened for writing.
 *
 * \throws config_file_write_exception
 *    Thrown when something prevents output of the file.
 *
 * \param config    The \c config_block object to output.
 * \param file_path The path to output the file to.
 */
void VITAL_CONFIG_EXPORT write_config_file( config_block_sptr const&  config,
                                            config_path_t const&      file_path );


// ------------------------------------------------------------------
/// Output to file the given \c config_block object to the specified stream.
/**
 * This function writes the specified config block to the specified
 * stream.  If a key has an associated description, it will be written
 * as a comment.  The key and value strings are written in a format
 * that can be read by the read_config_file() function or it can be
 * displayed.
 *
 * \throws config_file_write_exception
 *    Thrown when something prevents output of the file.
 *
 * \param config    The \c config_block object to output.
 * \param str       The output stream.
 */
void VITAL_CONFIG_EXPORT write_config( config_block_sptr const& config,
                                       std::ostream&            str );

} }

#endif // KWIVER_CONFIG_BLOCK_IO_H_
