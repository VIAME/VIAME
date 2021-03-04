// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VITAL Exceptions pertaining to IO operations
 */

#ifndef VITAL_CORE_EXCEPTIONS_IO_H
#define VITAL_CORE_EXCEPTIONS_IO_H

#include "base.h"
#include <string>
#include <vital/vital_types.h>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// VITAL Generic IO exception
class VITAL_EXCEPTIONS_EXPORT io_exception
  : public vital_exception
{
public:
  /// Constructor
  io_exception() noexcept;
  /// Destructor
  virtual ~io_exception() noexcept;
};

// ------------------------------------------------------------------
/// Exception for when a given path doesn't point to anything on the filesystem
class VITAL_EXCEPTIONS_EXPORT path_not_exists
  : public io_exception
{
public:
  /// Constructor
  /**
   * \param path The path that doesn't point to an existing file or directory
   */
  path_not_exists(path_t const& path) noexcept;
  /// Destructor
  virtual ~path_not_exists() noexcept;
};

// ------------------------------------------------------------------
/// Exception for when a given path doesn't point to a file.
class VITAL_EXCEPTIONS_EXPORT path_not_a_file
  : public io_exception
{
public:
  /// Constructor
  /**
   * \param path The path that doesn't point to a file.
   */
  path_not_a_file(path_t const& path) noexcept;
  /// Destructor
  virtual ~path_not_a_file() noexcept;
};

// ------------------------------------------------------------------
/// Exception for when a given path doesn't point to a directory.
class VITAL_EXCEPTIONS_EXPORT path_not_a_directory
  : public io_exception
{
public:
  /// Constructor
  /**
   * \param path The path that doesn't point to a directory.
   */
  path_not_a_directory(path_t const& path) noexcept;
  /// Destructor
  virtual ~path_not_a_directory() noexcept;
};

// ------------------------------------------------------------------
/// Exception for an encounter with an invalid file by some metric.
class VITAL_EXCEPTIONS_EXPORT invalid_file
  : public io_exception
{
public:
  /// Constructor
  /*
   * \param file    The file that has been deemed invalid
   * \param reason  The reason for invalidity.
   */
  invalid_file(path_t const& file, std::string const& reason) noexcept;
  /// Destructor
  virtual ~invalid_file() noexcept;
};

// ------------------------------------------------------------------
/// Exception for an encounter with invalid data by some metric
class VITAL_EXCEPTIONS_EXPORT invalid_data
  : public io_exception
{
public:
  /// Constructor
  invalid_data(std::string const& reason) noexcept;
  /// Destructor
  virtual ~invalid_data() noexcept;
};

// ------------------------------------------------------------------
/// Exception for when a file could not be found
class VITAL_EXCEPTIONS_EXPORT file_not_found_exception
  : public io_exception
{
public:
  /// Constructor
  /**
   * \param file_path The file path that was looked for.
   * \param reason    The reason the file wasn't found.
   */
  file_not_found_exception( path_t const& file_path, std::string const& reason ) noexcept;
  /// Deconstructor
  virtual ~file_not_found_exception() noexcept;
};

// ------------------------------------------------------------------
/// Exception for when a file could not be read for whatever reason.
class VITAL_EXCEPTIONS_EXPORT file_not_read_exception
  : public io_exception
{
public:
  ///Constructor
  /**
   * \param file_path The file path on which the read was attempted.
   * \param reason    The reason for the read exception.
   */
  file_not_read_exception( path_t const& file_path, std::string const& reason ) noexcept;
  /// Deconstructor
  virtual ~file_not_read_exception() noexcept;
};

// ------------------------------------------------------------------
/// Exception for when a file was not able to be written
class VITAL_EXCEPTIONS_EXPORT file_write_exception
  : public io_exception
{
public:
  /// Constructor
  /**
   * \param file_path The file path to which the write was attempted.
   * \param reason    The reason for the exception
   */
  file_write_exception( path_t const& file_path, std::string const& reason ) noexcept;
  /// Deconstructor
  virtual ~file_write_exception() noexcept;
};

} } // end vital namespace

#endif // VITAL_CORE_EXCEPTIONS_IO_H
