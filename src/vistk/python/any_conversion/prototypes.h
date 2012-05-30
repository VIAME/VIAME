/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_ANY_CONVERSION_PROTOTYPES_H
#define VISTK_PYTHON_ANY_CONVERSION_PROTOTYPES_H

#include "registration.h"

#include <vistk/python/util/python_gil.h>

#include <boost/python/converter/registry.hpp>
#include <boost/python/extract.hpp>

#include <Python.h>

/**
 * \file any_conversion/prototypes.h
 *
 * \brief Prototype functions for converting types to and from boost::any for Python.
 */

namespace vistk
{

namespace python
{

/**
 * \brief Converts a Python object into a boost::any.
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

  using namespace boost::python;

  extract<T> const ex(obj);
  if (ex.check())
  {
    new (storage) boost::any(ex());
    return true;
  }

  extract<T const> const exc(obj);
  if (exc.check())
  {
    new (storage) boost::any(exc());
    return true;
  }

  extract<T const&> const excr(obj);
  if (excr.check())
  {
    new (storage) boost::any(excr());
    return true;
  }

  return false;
}

/**
 * \brief Converts a boost::any into a Python object.
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

  using namespace boost::python;

  try
  {
    T const t = boost::any_cast<T>(any);
    object const o(t);
    return incref(o.ptr());
  }
  catch (boost::bad_any_cast&)
  {
  }

  return opt_pyobject_t();
}

/**
 * \brief Registers a type for conversion between Python and boost::any.
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

#endif // VISTK_PYTHON_ANY_CONVERSION_PROTOTYPES_H
