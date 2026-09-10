// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PYTHON_UTIL_PYSTREAM_H
#define SPROKIT_PYTHON_UTIL_PYSTREAM_H

#include <python/kwiver/sprokit/util/sprokit_python_util_export.h>

#include <pybind11/pybind11.h>

#include <istream>
#include <ostream>
#include <streambuf>
#include <vector>

namespace sprokit {

namespace python {

// ----------------------------------------------------------------------------
/// Stream buffer for reading from a Python file-like object
class SPROKIT_PYTHON_UTIL_EXPORT pyistreambuf
  : public std::streambuf
{
public:
  pyistreambuf( pybind11::object const& obj, std::size_t buffer_size = 1024 );
  ~pyistreambuf();

protected:
  int_type underflow() override;

private:
  pybind11::object m_obj;
  std::vector< char > m_buffer;
};

/// Input stream wrapping a Python file-like object
class SPROKIT_PYTHON_UTIL_EXPORT pyistream
  : public std::istream
{
public:
  pyistream( pybind11::object const& obj );
  ~pyistream();

private:
  pyistreambuf m_buf;
};

// ----------------------------------------------------------------------------
/// Stream buffer for writing to a Python file-like object
class SPROKIT_PYTHON_UTIL_EXPORT pyostreambuf
  : public std::streambuf
{
public:
  pyostreambuf( pybind11::object const& obj, std::size_t buffer_size = 1024 );
  ~pyostreambuf();

protected:
  int_type overflow( int_type ch ) override;
  int sync() override;

private:
  bool flush_buffer();

  pybind11::object m_obj;
  std::vector< char > m_buffer;
};

/// Output stream wrapping a Python file-like object
class SPROKIT_PYTHON_UTIL_EXPORT pyostream
  : public std::ostream
{
public:
  pyostream( pybind11::object const& obj );
  ~pyostream();

private:
  pyostreambuf m_buf;
};

} // namespace python

} // namespace sprokit

#endif // SPROKIT_PYTHON_UTIL_PYSTREAM_H
