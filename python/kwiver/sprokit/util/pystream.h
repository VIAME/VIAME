/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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
