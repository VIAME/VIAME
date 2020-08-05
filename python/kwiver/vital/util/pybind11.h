/*ckwg +29
 * Copyright 2018-2019 by Kitware, Inc.
 * Copyright 2016 by Wenzel Jakob
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

#ifndef VITAL_PYTHON_UTIL_PYBIND11_H
#define VITAL_PYTHON_UTIL_PYBIND11_H

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital {
namespace python {

/*
 * NOTE: These classes and macros have just been lifted from pybind11 and
 * rewritten to use the Python interpreter's standard PyGILState_* API, because
 * of the inherent compatibility issues in the pybind11 version. In the future,
 * we would like to get similar classes and macros merged into upstream pybind11
 * so that they become standard.
 */

class gil_scoped_acquire
{
  public:
    gil_scoped_acquire()
    {
      state = PyGILState_Ensure();
    }

    ~gil_scoped_acquire()
    {
      PyGILState_Release( state );
    }

  private:
    PyGILState_STATE state;
};

class gil_scoped_release
{
  public:
    gil_scoped_release()
    {
      state = PyEval_SaveThread();
    }

    ~gil_scoped_release()
    {
      PyEval_RestoreThread( state );
    }

  private:
    PyThreadState* state;
};

#define VITAL_PYBIND11_OVERLOAD_INT( ret_type, cname, name, ... ) \
{ \
    kwiver::vital::python::gil_scoped_acquire gil; \
    pybind11::function overload = pybind11::get_overload( static_cast< const cname* > ( this ), name ); \
    if( overload ) \
    { \
          auto o = overload( __VA_ARGS__ ); \
          if( pybind11::detail::cast_is_temporary_value_reference< ret_type >::value ) \
          { \
                  static pybind11::detail::overload_caster_t< ret_type > caster; \
                  return pybind11::detail::cast_ref< ret_type > ( std::move( o ), caster ); \
                } \
          else \
          { \
                  return pybind11::detail::cast_safe< ret_type > ( std::move( o ) ); \
                } \
        } \
}

#define VITAL_PYBIND11_OVERLOAD_NAME( ret_type, cname, name, fn, ... ) \
    VITAL_PYBIND11_OVERLOAD_INT( ret_type, cname, name, __VA_ARGS__ ) \
  return cname::fn( __VA_ARGS__ )

#define VITAL_PYBIND11_OVERLOAD_PURE_NAME( ret_type, cname, name, fn, ... ) \
    VITAL_PYBIND11_OVERLOAD_INT( ret_type, cname, name, __VA_ARGS__ ) \
  pybind11::pybind11_fail( "Tried to call pure virtual function \"" #cname "::" name "\"" );

#define VITAL_PYBIND11_OVERLOAD( ret_type, cname, fn, ... ) \
    VITAL_PYBIND11_OVERLOAD_NAME( ret_type, cname, #fn, fn, __VA_ARGS__ )

#define VITAL_PYBIND11_OVERLOAD_PURE( ret_type, cname, fn, ... ) \
    VITAL_PYBIND11_OVERLOAD_PURE_NAME( ret_type, cname, #fn, fn, __VA_ARGS__ )

} } }

#endif // VITAL_PYTHON_UTIL_PYBIND11_H
