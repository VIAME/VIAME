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
