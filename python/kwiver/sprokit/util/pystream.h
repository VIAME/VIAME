// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PYTHON_UTIL_PYSTREAM_H
#define SPROKIT_PYTHON_UTIL_PYSTREAM_H

#include <python/kwiver/sprokit/util/sprokit_python_util_export.h>

#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>
#include <pybind11/pybind11.h>

#include <iosfwd>

namespace sprokit {
namespace python {

class SPROKIT_PYTHON_UTIL_EXPORT pyistream_device
  : public boost::iostreams::source
{
  public:
    pyistream_device(pybind11::object const& obj);
    ~pyistream_device();

    std::streamsize read(char_type* s, std::streamsize n);

  private:
    pybind11::object m_obj;
};

typedef boost::iostreams::stream<pyistream_device> pyistream;

// ----------------------------------------------------------------------------
class SPROKIT_PYTHON_UTIL_EXPORT pyostream_device
  : public boost::iostreams::sink
{
  public:
    pyostream_device(pybind11::object const& obj);
    ~pyostream_device();

    std::streamsize write(char_type const* s, std::streamsize n);

  private:
    pybind11::object m_obj;
};

typedef boost::iostreams::stream<pyostream_device> pyostream;

}
}

#endif // SPROKIT_PYTHON_UTIL_PYSTREAM_H
