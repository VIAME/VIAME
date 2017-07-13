/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
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

#ifndef SPROKIT_PYTHON_UTIL_PYTHON_WRAP_CONST_SHARED_PTR_H
#define SPROKIT_PYTHON_UTIL_PYTHON_WRAP_CONST_SHARED_PTR_H

#include <boost/python/pointee.hpp>
#include <boost/get_pointer.hpp>

#include <memory>

// Retrieved from http://mail.python.org/pipermail/cplusplus-sig/2006-November/011329.html
namespace boost
{

namespace python
{

template <typename T>
inline
T*
get_pointer(boost::shared_ptr<T const> const& p)
{
  return const_cast<T*>(p.get());
}

template <typename T>
struct pointee<boost::shared_ptr<T const> >
{
  typedef T type;
};

// Don't hide other get_pointer instances.
using boost::python::get_pointer;
using boost::get_pointer;

}

}

#endif // SPROKIT_PYTHON_UTIL_PYTHON_WRAP_CONST_SHARED_PTR_H
