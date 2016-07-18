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
 * Refer to \ref config_file_format "config file format" for more
 * information on the file entries.
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
 * This method reads the specified config file and returns the
 * resulting config block. If a search path is supplied, any files
 * included by config files are resolved using that list if they do
 * not have an absolute path.
 *
 * \throws config_file_not_found_exception
 *    Thrown when the file could not be found on the file system.
 * \throws config_file_not_read_exception
 *    Thrown when the file could not be read or parsed for whatever reason.
 *
 * \param file_path   The path to the file to read in.
 * \param search_path An optional list of directories to use in locating included files.
 *
 * \return A \c config_block object representing the contents of the read-in file.
 */
config_block_sptr VITAL_CONFIG_EXPORT read_config_file(
  config_path_t const&      file_path,
  config_path_list_t const& search_path = config_path_list_t() );


// ------------------------------------------------------------------
/// Read in (a) configuration file(s), producing a \c config_block object
/**
 * This function reads one or more configuration files from a search
 * path. The search path is based on environment variables, system
 * defaults, and application defaults. More on this later.
 *
 * The config reader tries to locate the specified config file using
 * the search path. If the file is not found, an exception is
 * thrown. If the file is located and the \c merge parameter is \b
 * true (default value), then the remaining directories in the search
 * path are checked to see if additional versions of the file can be
 * found. If so, then the contents are merged, in the order found,
 * into the resulting config block. If the \c merge parameter is \b
 * false. then reading process stops after the first file is found.
 *
 * A platform specific search path is constructed as follows:
 *
 * ## Windows Platform
 * - ${KWIVER_CONFIG_PATH}          (if set)
 * - $<CSIDL_LOCAL_APPDATA>/<app-name>[/<app-version>]/config
 * - $<CSIDL_APPDATA>/<app-name>[/<app-version>]/config
 * - $<CSIDL_COMMON_APPDATA>/<app-name>[/<app-version>]/config
 * - <install-dir>/share/<app-name>[/<app-version>]/config
 * - <install-dir>/share/config
 * - <install-dir>/config
 *
 * ## OS/X Apple Platform
 * - ${KWIVER_CONFIG_PATH}                                    (if set)
 * - ${XDG_CONFIG_HOME}/<app-name>[/<app-version>]/config     (if $XDG_CONFIG_HOME set)
 * - ${HOME}/.config/<app-name>[/<app-version>]/config        (if $HOME set)
 * - /etc/xdg/<app-name>[/<app-version>]/config
 * - /etc/<app-name>[/<app-version>]/config
 * - ${HOME}/Library/Application Support/<app-name>[/<app-version>]/config (if $HOME set)
 * - /Library/Application Support/<app-name>[/<app-version>]/config
 * - /usr/local/share/<app-name>[/<app-version>]/config
 * - /usr/share/<app-name>[/<app-version>]/config
 *
 * If <install-dir> is not `/usr` or `/usr/local`:
 *
 * - <install-dir>/share/<app-name>[/<app-version>]/config
 * - <install-dir>/share/config
 * - <install-dir>/config
 * - <install-dir>/Resources/config
 *
 * ## Other Posix Platforms (e.g. Linux)
 * - ${KWIVER_CONFIG_PATH}                                    (if set)
 * - ${XDG_CONFIG_HOME}/<app-name>[/<app-version>]/config     (if $XDG_CONFIG_HOME set)
 * - ${HOME}/.config/<app-name>[/<app-version>]/config        (if $HOME set)
 * - /etc/xdg/<app-name>[/<app-version>]/config
 * - /etc/<app-name>[/<app-version>]/config
 * - /usr/local/share/<app-name>[/<app-version>]/config
 * - /usr/share/<app-name>[/<app-version>]/config
 *
 * If <install-dir> is not `/usr` or `/usr/local`:
 *
 * - <install-dir>/share/<app-name>[/<app-version>]/config
 * - <install-dir>/share/config
 * - <install-dir>/config
 *
 * The environment variable \c KWIVER_CONFIG_PATH can be set with a
 * list of one or more directories, in the same manner as the native
 * execution \c PATH variable, to be searched for config files.
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
 *   \c false, read only the first matching file. If this parameter is omitted
 *   the configs are merged.
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

config_path_list_t VITAL_CONFIG_EXPORT
config_file_paths( std::string const& application_name,
                   std::string const& application_version,
                   config_path_t const& install_prefix );

} }

#endif // KWIVER_CONFIG_BLOCK_IO_H_
