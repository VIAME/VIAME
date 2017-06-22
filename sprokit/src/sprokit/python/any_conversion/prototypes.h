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

#ifndef SPROKIT_PYTHON_ANY_CONVERSION_PROTOTYPES_H
#define SPROKIT_PYTHON_ANY_CONVERSION_PROTOTYPES_H

#include "registration.h"

#include <sprokit/python/util/python_gil.h>

#include <boost/python/converter/registry.hpp>
#include <boost/python/extract.hpp>

#include <sprokit/python/util/python.h>

/**
 * \file any_conversion/prototypes.h
 *
 * \brief Prototype functions for converting types to and from boost::any for Python.
 */

namespace sprokit
{

namespace python
{

/**
 * \brief Convert a Python object into a boost::any.
 *
 * \param obj The object to convert.
 * \param storage The memory location to construct the object.
 *
 * \return True if the conversion succeeded, false otherwise.
 */
template <typename T>
bool
from_prototype(PyObject* obj, void* storage)
{
  python_gil const gil;

  (void)gil;

  boost::python::extract<T> const ex(obj);
  if (ex.check())
  {
    new (storage) boost::any(ex());
    return true;
  }

  boost::python::extract<T const> const exc(obj);
  if (exc.check())
  {
    new (storage) boost::any(exc());
    return true;
  }

  boost::python::extract<T const&> const excr(obj);
  if (excr.check())
  {
    new (storage) boost::any(excr());
    return true;
  }

  return false;
}


/**
 * \brief Convert a boost::any into a Python object.
 *
 * \param any The object to convert.
 *
 * \return The object if created, nothing otherwise.
 */
template <typename T>
opt_pyobject_t
to_prototype(boost::any const& any)
{
  python_gil const gil;

  (void)gil;

  try
  {
    T const t = boost::any_cast<T>(any);
    boost::python::object const o(t);
    return boost::python::incref(o.ptr());
  }
  catch (boost::bad_any_cast const&)
  {
  }

  // Returning a default object indicates failure
  return opt_pyobject_t();
}


/**
 * \brief Register a type for conversion between Python and boost::any.
 *
 * \param priority The priority for the type conversion.
 */
template <typename T>
void
register_type(priority_t priority)
{
  register_conversion(priority, from_prototype<T>, to_prototype<T>);
}

}

}

#endif // SPROKIT_PYTHON_ANY_CONVERSION_PROTOTYPES_H
