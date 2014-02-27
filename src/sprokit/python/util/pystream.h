/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PYTHON_UTIL_PYSTREAM_H
#define SPROKIT_PYTHON_UTIL_PYSTREAM_H

#include "util-config.h"

#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/python/object.hpp>

#include <iosfwd>

namespace sprokit
{

namespace python
{

class SPROKIT_PYTHON_UTIL_EXPORT pyistream_device
  : public boost::iostreams::source
{
  public:
    pyistream_device(boost::python::object const& obj);
    ~pyistream_device();

    std::streamsize read(char_type* s, std::streamsize n);
  private:
    boost::python::object m_obj;
};

typedef boost::iostreams::stream<pyistream_device> pyistream;

class SPROKIT_PYTHON_UTIL_EXPORT pyostream_device
  : public boost::iostreams::sink
{
  public:
    pyostream_device(boost::python::object const& obj);
    ~pyostream_device();

    std::streamsize write(char_type const* s, std::streamsize n);
  private:
    boost::python::object m_obj;
};

typedef boost::iostreams::stream<pyostream_device> pyostream;

}

}

#endif // SPROKIT_PYTHON_UTIL_PYSTREAM_H
