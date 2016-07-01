/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * \brief C Interface to \p vital::config_block class
 */

#ifndef VITAL_C_CONFIG_BLOCK_H_
#define VITAL_C_CONFIG_BLOCK_H_


#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/common.h>
#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/error_handle.h>

#include <stdbool.h>


/// Structure for opaque pointers to \p config_block objects
typedef struct vital_config_block_s vital_config_block_t;


// Config block constant getters

/// Separator between blocks within the config
/**
 * Pointer returned is cached, and the same pointer is returned every call.
 */
VITAL_C_EXPORT
char const*
vital_config_block_block_sep();

/// The magic group for global parameters
/**
 * Pointer returned is cached, and the same pointer is returned every call.
 */
VITAL_C_EXPORT
char const*
vital_config_block_global_value();


/// Create a new, empty \p config_block object
/**
 * \return Opaque pointer to an empty config_block with the default name, or 0
 *         if construction failed.
 */
VITAL_C_EXPORT
vital_config_block_t*
vital_config_block_new();

/// Create a new, empty \p config_block object with a name
/**
 * \param name String name for the constructed config block.
 * \return Opaque pointer to an empty config_block with the default name, or 0
 *         if construction failed.
 */
VITAL_C_EXPORT
vital_config_block_t*
vital_config_block_new_named( char const *name );

/// Destroy a config block object
/**
 * \param cb Opaque pointer to config_block instance.
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_config_block_destroy( vital_config_block_t *cb,
                            vital_error_handle_t *eh );

/// Get the name of the \p config_block instance
/**
 * \param cb Opaque pointer to config_block instance.
 * \param eh Vital error handle instance
 * \return String name of the given config_block.
 */
VITAL_C_EXPORT
char const*
vital_config_block_get_name( vital_config_block_t *cb,
                             vital_error_handle_t *eh );

/// Get a subblock from the configuration.
/**
 * Retrieve an unlinked configuration subblock from the current
 * configuration. Changes made to it do not affect \p *cb.
 *
 * \param cb Opaque pointer to the config_block instance
 * \param key The name of the sub-configuration to retrieve.
 * \param eh Vital error handle instance
 * \return Pointer to a new config_block instance with copies of values.
 */
VITAL_C_EXPORT
vital_config_block_t*
vital_config_block_subblock( vital_config_block_t *cb,
                             char const *key,
                             vital_error_handle_t *eh );

/// Get a subblock view into the configuration.
/**
 * Retrieve a view into the current configuration. Changes made to \c *cb
 * are seen through the view and vice versa.
 *
 * \param cb Opaque pointer to a config_block instance.
 * \param key The name of the sub-configuration to retrieve.
 * \param eh Vital error handle instance
 * \return A subblock which linkes to \p *cb.
 */
VITAL_C_EXPORT
vital_config_block_t*
vital_config_block_subblock_view( vital_config_block_t *cb,
                                  char const *key,
                                  vital_error_handle_t *eh );

/// Get the string value for a key
/**
 * This may fail if the key given doesn't exist, populating \c eh with error
 * code 1.
 *
 * \param cb Opaque pointer to a config_block instance.
 * \param key The index of the configuration value to retrieve.
 * \return A new copy of the string value stored within the configuration. This
 *   should be free'd when done with the return value.
 */
VITAL_C_EXPORT
char const*
vital_config_block_get_value( vital_config_block_t *cb,
                              char const *key,
                              vital_error_handle_t *eh );

/// Get the boolean value for a key
/**
 * This may fail if the key given doesn't exist, populating \c eh with error
 * code 1.
 *
 * \param cb Opaque pointer to a config_block instance.
 * \param key The index of the configuration value to retrieve.
 * \return The boolean value stored within the configuration.
 */
VITAL_C_EXPORT
bool
vital_config_block_get_value_bool( vital_config_block_t *cb,
                                   char const *key,
                                   vital_error_handle_t *eh );

/// Get the string value for a key if it exists, else the default
/**
 * \param cb Opaque pointer to a config_block instance.
 * \param key The index of the configuration value to retrieve.
 * \param deflt A default value to return if the given key does not have an
 *              associated value.
 * \return A new copy of the string value stored within the configuration,
 *   otherwise a copy of \c deflt is returned. This should be free'd when done
 *   with the return value.
 */
VITAL_C_EXPORT
char const*
vital_config_block_get_value_default( vital_config_block_t *cb,
                                      char const *key,
                                      char const *deflt,
                                      vital_error_handle_t *eh );

/// Get the boolean value for a key if it exists, else the default
/**
 * \param cb Opaque pointer to a config_block instance.
 * \param key The index of the configuration value to retrieve.
 * \param deflt A default value to return if the given key does not have an
 *              associated value.
 * \return the bool value stored within the configuration.
 */
VITAL_C_EXPORT
bool
vital_config_block_get_value_default_bool( vital_config_block_t *cb,
                                           char const *key,
                                           bool deflt,
                                           vital_error_handle_t *eh );

/// Get the description associated to a value
/**
 * If the provided key exists but has no description associated with it, an
 * empty string is returned.
 *
 * This may fail if the key given doesn't exist, populating \c eh with error
 * code 1.
 *
 * \param cb Opaque pointer to a config_block instance.
 * \param key The name of the parameter to get the description of.
 * \returns A copy of the string description of the give key or NULL if the key
 *   was not found. When not NULL, this should be free'd when done with the
 *   return value.
 */
VITAL_C_EXPORT
char const*
vital_config_block_get_description( vital_config_block_t *cb,
                                    char const *key,
                                    vital_error_handle_t *eh);

/// Set a string value within the configuration.
/**
 * If the key is marked read only, the error handle is populated with error code
 * 1.
 *
 * If this key already exists, has a description and no new description
 * was passed with this \c set_value call, the previous description is
 * retained. We assume that the previous description is still valid and
 * this is a value overwrite. If it is intended for the description to also
 * be overwritten, an \c unset_value call should be performed on the key
 * first, and then this \c set_value call.
 *
 * \param cb Opaque pointer to a config_block instance.
 * \param key The index of the configuration value to set.
 * \param value The value to set for the \p key.
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_config_block_set_value( vital_config_block_t *cb,
                              char const *key,
                              char const *value,
                              vital_error_handle_t *eh );

/// Set a string value with an associated description
/**
 * If the key is marked read only, the error handle is populated with error code
 * 1.
 *
 * If this key already exists, has a description and no new description
 * was passed with this \c set_value call, the previous description is
 * retained. We assume that the previous description is still valid and
 * this is a value overwrite.
 *
 * \param cb Opaque pointer to a config_block instance.
 * \param key The index of the configuration value to set.
 * \param value The value to set for the \p key.
 * \param descr Description of the key, overriding any existing description for
 *   the given key.
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_config_block_set_value_descr( vital_config_block_t *cb,
                                    char const *key,
                                    char const *value,
                                    char const *description,
                                    vital_error_handle_t *eh );

/// Remove a key/value pair from the configuration.
/**
 * If the provided key is marked as read-only, the error handle is given code 1.
 * If the requested key does not exist, the error handle is given code 2.
 *
 * \param cb Opaque pointer to a config_block instance.
 * \param key The index of the configuration value to set.
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_config_block_unset_value( vital_config_block_t *cb,
                                char const *key,
                                vital_error_handle_t *eh );

/// Query if a value is read-only
/**
 * If the requested key does not exist, \c false is returned.
 *
 * \param cb Opaque pointer to a config_block instance.
 * \param key The key to check.
 * \param eh Vital error handle instance
 * \returns True if the \p key has been marked as read-only, and false
 *          otherwise.
 */
VITAL_C_EXPORT
bool
vital_config_block_is_read_only( vital_config_block_t *cb,
                                 char const *key,
                                 vital_error_handle_t *eh );

/// Mark the given key as read-only
/**
 * This provided key is marked as read-only even if it doesn't currently
 * exist in the given config_block instance.
 *
 * \param cb Opaque pointer to a config_block instance.
 * \param key The key to mark as read-only.
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_config_block_mark_read_only( vital_config_block_t *cb,
                                   char const *key,
                                   vital_error_handle_t *eh );

/// Merge the values in \p other into the current config \p cb.
/**
 * If the entry in this config is marked as
 * read-only, error code 1 is set in the handler and the merge operation is left
 * partially complete. If an entry in the specified config block is marked as
 * read-only, that attribute is *not* copied to this block.
 *
 * \note Any values currently set within \c *this will be overwritten if
 *       conflicts occur.
 *
 * \param cb Opaque pointer to a config_block instance.
 * \param other Opaque pointer to a config_block instance whose key/value
 *              pairs are to be merged into \p cb.
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_config_block_merge_config( vital_config_block_t *cb,
                                 vital_config_block_t *other,
                                 vital_error_handle_t *eh );

/// Check if a value exists for the given key
/**
 * \param cb Opaque pointer to a config_block instance.
 * \param key The index of the configuration value to check.
 * \param eh Vital error handle instance
 * \return true if \p cb has a value for the given \p key, else false.
 */
VITAL_C_EXPORT
bool
vital_config_block_has_value( vital_config_block_t *cb,
                              char const *key,
                              vital_error_handle_t *eh );

/// Return the values available in the configuration.
/**
 * We are expecting that the \p length and \p keys parameters will be passed
 * by reference by the user as they are dereferenced within the function for
 * value assignment.
 *
 * \param[in] cb Opaque pointer to a config_block instance.
 * \param[out] length The number of available keys in \p cb.
 * \param[out] keys Pointer to an array of char* strings.
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_config_block_available_values( vital_config_block_t *cb,
                                     unsigned int *length,
                                     char ***keys,
                                     vital_error_handle_t *eh );


/// Read in a configuration file, producing a config_block object
/**
 * This may fail if the given filepath is not found, could not be read, or some
 * other filesystem error. In such a case, a NULL pointer is returned and the
 * given error handle, if non-null, will be given an error code and message.
 *
 * Error Codes:
 *  (0) Successful read
 *  (1) File whose path was given could not be found.
 *  (2) File whose path was given could not be read.
 *  (3) File whose path was given could not be parsed.
 *  (-1) Some other exception occurred
 *
 * \param filepath   The path to the file to read in.
 * \return A an object representing the contents of the read-in file.
 */
VITAL_C_EXPORT
vital_config_block_t*
vital_config_block_file_read( char const *filepath,
                              vital_error_handle_t *eh );


/// Read in a configuration file from standard locations, producing a config_block object
/**
 * This may fail if the given name is not found in any of the standard
 * directories, or if an error occurs while trying to read one of the located
 * files. In such a case, a NULL pointer is returned and the given error
 * handle, if non-null, will be given an error code and message.
 *
 * Error Codes:
 *  (0) Successful read
 *  (1) No matching file was found.
 *  (2) A matching file could not be read.
 *  (3) A matching file could not be parsed.
 *  (-1) Some other exception occurred
 *
 * \param name
 *   The name to the file(s) to read in.
 * \param application_name
 *   The application name, used to build the list of standard locations to be
 *   searched.
 * \param application_version
 *   The application version number, used to build the list of standard
 *   locations to be searched. Pass \c NULL to skip versioned paths.
 * \param install_prefix
 *   The prefix to which the application is installed (should be one directory
 *   higher than the location of the executing binary). Pass \c NULL to skip
 *   searching a non-standard prefix.
 * \param merge
 *   If \c true, search all locations for matching config files, merging their
 *   contents, with files earlier in the search order taking precedence. If
 *   \c false, read only the first matching file.
 *
 * \return A object representing the contents of the read-in file.
 */
VITAL_C_EXPORT
vital_config_block_t*
vital_config_block_file_read_from_standard_location(
  char const*           name,
  char const*           application_name,
  char const*           application_version,
  char const*           install_prefix,
  bool                  merge,
  vital_error_handle_t* eh );


/// Output to file the given \c config_block object to the specified file path
/**
 * If a file exists at the target location, it will be overwritten. If the
 * containing directory of the given path does not exist, it will be created
 * before the file is opened for writing.
 *
 * This may fail if the given filepath is not found, could not be read, or some
 * other filesystem error. In such a case, a NULL pointer is returned and the
 * given error handle, if non-null, will be given an error code and message.
 *
 * Error Codes:
 *  (0) Successful write.
 *  (1) Exception occurred when writing file
 *  (-1) Some other exception occurred
 */
VITAL_C_EXPORT
void
vital_config_block_file_write( vital_config_block_t *cb,
                               char const *filepath,
                               vital_error_handle_t *eh );


#ifdef __cplusplus
}
#endif


#endif // VITAL_C_CONFIG_BLOCK_H_
