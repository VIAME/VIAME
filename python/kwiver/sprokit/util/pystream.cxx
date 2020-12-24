// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pystream.h"

#include <pybind11/pybind11.h>

#include <python/kwiver/vital/util/pybind11.h>

#include <algorithm>
#include <string>

#include <cstddef>

namespace sprokit
{

namespace python
{

pyistream_device
::pyistream_device(pybind11::object const& obj)
  : m_obj(obj)
{
  // \todo Check that the object has a "read" attribute and that it is callable.
}

pyistream_device
::~pyistream_device()
{
}

std::streamsize
pyistream_device
::read(char_type* s, std::streamsize n)
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  pybind11::str const bytes = pybind11::str(m_obj.attr("read")(n));

  pybind11::ssize_t const sz = len(bytes);

  if (sz)
  {
    std::string const cppstr = bytes.cast<std::string>();

    std::copy(cppstr.begin(), cppstr.end(), s);

    return sz;
  }
  else
  {
    return -1;
  }
}

pyostream_device
::pyostream_device(pybind11::object const& obj)
  : m_obj(obj)
{
  // \todo Check that the object has a "write" attribute and that it is callable.
}

pyostream_device
::~pyostream_device()
{
}

std::streamsize
pyostream_device
::write(char_type const* s, std::streamsize n)
{
  kwiver::vital::python::gil_scoped_acquire acquire;
  (void)acquire;

  pybind11::str const bytes(s, static_cast<size_t>(n));

  m_obj.attr("write")(bytes);

  return n;
}

}

}
