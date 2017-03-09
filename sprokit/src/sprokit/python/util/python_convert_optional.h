/*ckwg +29
 * Copyright 2011-2012 by Kitware, Inc.
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

#ifndef SPROKIT_PYTHON_UTIL_PYTHON_CONVERT_OPTIONAL_H
#define SPROKIT_PYTHON_UTIL_PYTHON_CONVERT_OPTIONAL_H

#include "python_gil.h"

#include <boost/python/converter/registry.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/to_python_converter.hpp>
#include <boost/optional.hpp>

#include <sprokit/python/util/python.h>

/**
 * \file python_convert_optional.h
 *
 * \brief Helpers for working with boost::optional in Python.
 */

namespace sprokit
{

namespace python
{

template <typename T>
class boost_optional_converter
{
  public:
    typedef T type_t;
    typedef boost::optional<T> optional_t;

    boost_optional_converter();
    ~boost_optional_converter();

    static void* convertible(PyObject* obj);
    static PyObject* convert(optional_t const& opt);
    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data);
};

template <typename T>
boost_optional_converter<T>
::boost_optional_converter()
{
  boost::python::converter::registry::push_back(
    &convertible,
    &construct,
    boost::python::type_id<optional_t>());
}

template <typename T>
boost_optional_converter<T>
::~boost_optional_converter()
{
}

template <typename T>
void*
boost_optional_converter<T>
::convertible(PyObject* obj)
{
  python::python_gil const gil;

  (void)gil;

  if (obj == Py_None)
  {
    return obj;
  }

  using namespace boost::python::converter;

  registration const& converters(registered<type_t>::converters);

  if (implicit_rvalue_convertible_from_python(obj, converters))
  {
    rvalue_from_python_stage1_data data = rvalue_from_python_stage1(obj, converters);
    return rvalue_from_python_stage2(obj, data, converters);
  }

  return NULL;
}

template <typename T>
PyObject*
boost_optional_converter<T>
::convert(optional_t const& opt)
{
  python::python_gil const gil;

  (void)gil;

  if (opt)
  {
    return boost::python::to_python_value<T>()(*opt);
  }
  else
  {
    Py_RETURN_NONE;
  }
}

template <typename T>
void
boost_optional_converter<T>
::construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data)
{
  python::python_gil const gil;

  (void)gil;

  void* const storage = reinterpret_cast<boost::python::converter::rvalue_from_python_storage<optional_t>*>(data)->storage.bytes;

#define CONSTRUCT(args)            \
  do                               \
  {                                \
    new (storage) optional_t args; \
    data->convertible = storage;   \
    return;                        \
  } while (false)

  if (obj == Py_None)
  {
    CONSTRUCT(());
  }
  else
  {
    type_t* const t = reinterpret_cast<type_t*>(data->convertible);
    CONSTRUCT((*t));
  }

#undef CONSTRUCT
}

template <typename T>
void register_optional_converter(char const* name, char const* desc);

template <typename T>
class boost_optional_operations
{
  public:
    typedef T type_t;
    typedef boost::optional<T> optional_t;

    static bool not_(optional_t const& opt);
    static type_t& get(optional_t& opt);
    static type_t const& get_default(optional_t const& opt, type_t const& def);
};

template <typename T>
void
register_optional_converter(char const* name, char const* desc)
{
  typedef boost_optional_converter<T> converter_t;
  typedef typename converter_t::optional_t optional_t;
  typedef boost_optional_operations<T> operations_t;

  boost::python::class_<optional_t>(name
    , desc
    , boost::python::no_init)
    .def(boost::python::init<>())
    .def(boost::python::init<T>())
    .def("empty", &operations_t::not_
      , "True if there is no value, False otherwise.")
    .def("get", &operations_t::get
      , "Returns the contained value, or None if there isn\'t one."
      , boost::python::return_internal_reference<>())
    .def("get", &operations_t::get_default
      , (boost::python::arg("default"))
      , "Returns the contained value, or default if there isn\'t one."
      , boost::python::return_internal_reference<>())
  ;

  boost::python::to_python_converter<optional_t, converter_t>();
  converter_t();

  boost::python::implicitly_convertible<T, optional_t>();
}

template <typename T>
bool
boost_optional_operations<T>
::not_(optional_t const& opt)
{
  return !opt;
}

template <typename T>
T&
boost_optional_operations<T>
::get(optional_t& opt)
{
  return opt.get();
}

template <typename T>
T const&
boost_optional_operations<T>
::get_default(optional_t const& opt, T const& def)
{
  return opt.get_value_or(def);
}

}

}

#endif // SPROKIT_PYTHON_UTIL_PYTHON_CONVERT_OPTIONAL_H
