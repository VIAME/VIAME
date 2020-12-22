// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for IO exceptions
 */

#include "io.h"
#include <sstream>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
io_exception
::io_exception() noexcept
{
  m_what = "An IO exception occurred.";
}

io_exception
::~io_exception() noexcept
{
}

// ------------------------------------------------------------------
path_not_exists
::path_not_exists( path_t const& path ) noexcept
{
  std::ostringstream sstr;

  sstr << "Path does not exist: " << path;
  m_what = sstr.str();
}

path_not_exists
::~path_not_exists() noexcept
{
}

// ------------------------------------------------------------------
path_not_a_file
::path_not_a_file( path_t const& path ) noexcept
{
  m_what = "Path does not point to a file: " + path;
}

path_not_a_file
::~path_not_a_file() noexcept
{
}

// ------------------------------------------------------------------
path_not_a_directory
::path_not_a_directory( path_t const& path ) noexcept
{
  m_what = "Path does not point to a directory: " + path;
}

path_not_a_directory
::~path_not_a_directory() noexcept
{
}

// ------------------------------------------------------------------
invalid_file
::invalid_file( path_t const& path, std::string const& reason ) noexcept
{
  std::ostringstream ss;

  ss << "Invalid file " << path << ": " << reason;
  m_what = ss.str();
}

invalid_file
::~invalid_file() noexcept
{
}

// ------------------------------------------------------------------
invalid_data
::invalid_data( std::string const& reason ) noexcept
{
  m_what = "Invalid data: " + reason;
}

invalid_data
::~invalid_data() noexcept
{
}

// ------------------------------------------------------------------
file_not_found_exception
::file_not_found_exception( path_t const& file_path, std::string const& reason ) noexcept
{
  std::ostringstream sstr;

  sstr  << "Could not find file at location \'" << file_path << "\': "
        << reason;
  m_what = sstr.str();
}

file_not_found_exception
::~file_not_found_exception() noexcept
{
}

// ------------------------------------------------------------------
file_not_read_exception
::file_not_read_exception( path_t const& file_path, std::string const& reason ) noexcept
{
  std::ostringstream sstr;

  sstr  << "Failed to read from file \'" << file_path << "\': "
        << reason;
  m_what = sstr.str();
}

file_not_read_exception
::~file_not_read_exception() noexcept
{
}

// ------------------------------------------------------------------
file_write_exception
::file_write_exception( path_t const& file_path, std::string const& reason ) noexcept
{
  std::ostringstream sstr;

  sstr  << "Failed to write to file \'" << file_path << "\': "
        << reason;
  m_what = sstr.str();
}

file_write_exception
::~file_write_exception() noexcept
{
}

} }   // end vital namespace
