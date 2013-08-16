/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pystream.h"

#include "python_gil.h"

#include <boost/python/extract.hpp>
#include <boost/python/str.hpp>

#include <algorithm>
#include <string>

#include <cstddef>

namespace sprokit
{

namespace python
{

pyistream_device
::pyistream_device(boost::python::object const& obj)
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
  python::python_gil const gil;

  (void)gil;

  boost::python::str const bytes = boost::python::str(m_obj.attr("read")(n));

  boost::python::ssize_t const sz = boost::python::len(bytes);

  if (sz)
  {
    std::string const cppstr = boost::python::extract<std::string>(bytes);

    std::copy(cppstr.begin(), cppstr.end(), s);

    return sz;
  }
  else
  {
    return -1;
  }
}

pyostream_device
::pyostream_device(boost::python::object const& obj)
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
  python::python_gil const gil;

  (void)gil;

  boost::python::str const bytes(s, static_cast<size_t>(n));

  m_obj.attr("write")(bytes);

  return n;
}

}

}
