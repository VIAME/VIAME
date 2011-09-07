/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pystream.h"

#include <algorithm>
#include <string>

#include <cstddef>

pyistream_device
::pyistream_device(boost::python::object const& obj)
  : m_obj(obj)
{
}

pyistream_device
::~pyistream_device()
{
}

std::streamsize
pyistream_device
::read(char_type* s, std::streamsize n)
{
  boost::python::str bytes = boost::python::str(m_obj.attr("read")(n));

  long const sz = boost::python::len(bytes);

  if (sz)
  {
    std::string cppstr = boost::python::extract<std::string>(bytes);

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
}

pyostream_device
::~pyostream_device()
{
}

std::streamsize
pyostream_device
::write(char_type const* s, std::streamsize n)
{
  boost::python::str bytes(s, static_cast<size_t>(n));

  m_obj.attr("write")(bytes);

  return n;
}
