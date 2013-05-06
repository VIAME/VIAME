/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PYTHON_ANY_CONVERSION_REGISTRATION_H
#define SPROKIT_PYTHON_ANY_CONVERSION_REGISTRATION_H

#include "any_conversion-config.h"

#include <boost/any.hpp>
#include <boost/cstdint.hpp>
#include <boost/function.hpp>
#include <boost/optional.hpp>

#include <Python.h>

/**
 * \file any_conversion/registration.h
 *
 * \brief Helpers for working with boost::any in Python.
 */

namespace sprokit
{

namespace python
{

/// A type for a possible Python conversion.
typedef boost::optional<PyObject*> opt_pyobject_t;

/// A function which converts from Python, returning \c true on success.
typedef boost::function<bool (PyObject*, void*)> from_any_func_t;
/// A function which converts to Python, returning the object on success.
typedef boost::function<opt_pyobject_t (boost::any const&)> to_any_func_t;

/// A priority for converting between boost::any and Python.
typedef uint64_t priority_t;

/**
 * \brief Register functions for conversions between boost::any and Python.
 *
 * \param priority The priority for the type conversion.
 * \param from The function for converting from Python.
 * \param to The function for converting to Python.
 */
SPROKIT_PYTHON_ANY_CONVERSION_EXPORT void register_conversion(priority_t priority, from_any_func_t from, to_any_func_t to);

}

}

#endif // SPROKIT_PYTHON_ANY_CONVERSION_REGISTRATION_H
