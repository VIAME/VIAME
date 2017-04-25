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

#ifndef SPROKIT_PYTHON_ANY_CONVERSION_REGISTRATION_H
#define SPROKIT_PYTHON_ANY_CONVERSION_REGISTRATION_H

#include "any_conversion-config.h"

#include <boost/any.hpp>
#include <boost/cstdint.hpp>
#include <boost/function.hpp>
#include <boost/optional.hpp>

#include <sprokit/python/util/python.h>

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

/// A priority for converting between boost::any and Python.
typedef uint64_t priority_t;



// ==================================================================
/// A function which converts from Python, returning \c true on success.
typedef boost::function<bool (PyObject*, void*)> from_any_func_t;

/// A function which converts to Python, returning the object on success.
typedef boost::function<opt_pyobject_t (boost::any const&)> to_any_func_t;


/**
 * \brief Register functions for conversions between boost::any and Python.
 *
 * These conversion functions are registered with the sprokit python
 * bindings and the conversions are attempted in priority order until
 * one conversion succeeds.
 *
 * Note that multiple conversions may have the same priority value.
 *
 * Note that zero priority will be attempted first, so higher values
 * get tried later.
 *
 * The concept of priority seems useful, but there needs to be some
 * guidance on how best to use the full range.
 *
 *\sa register_type() is a convenience function to generate converters
 *and register them.
 *
 * \param priority The priority for the type conversion (zero being high).
 * \param from The function for converting from Python.
 * \param to The function for converting to Python.
 */
SPROKIT_PYTHON_ANY_CONVERSION_EXPORT
  void register_conversion(priority_t      priority,
                           from_any_func_t from,
                           to_any_func_t   to);

}

}

#endif // SPROKIT_PYTHON_ANY_CONVERSION_REGISTRATION_H
