/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_HELPERS_PYSTREAM_H
#define VISTK_PYTHON_HELPERS_PYSTREAM_H

#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/python/object.hpp>

#include <iostream>

class pyistream_device
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

class pyostream_device
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

#endif // VISTK_PYTHON_HELPERS_PYSTREAM_H
