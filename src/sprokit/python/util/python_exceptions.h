/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PYTHON_UTIL_PYTHON_EXCEPTIONS_H
#define SPROKIT_PYTHON_UTIL_PYTHON_EXCEPTIONS_H

#include "util-config.h"

namespace sprokit
{

namespace python
{

/// \todo More useful output?

#define SPROKIT_PYTHON_HANDLE_EXCEPTION(call)     \
  try                                             \
  {                                               \
    call;                                         \
  }                                               \
  catch (boost::python::error_already_set const&) \
  {                                               \
    sprokit::python::python_print_exception();    \
                                                  \
    throw;                                        \
  }

#define SPROKIT_PYTHON_IGNORE_EXCEPTION(call)      \
  try                                              \
  {                                                \
    call;                                          \
  }                                                \
  catch (boost::python::error_already_set const&)  \
  {                                                \
    sprokit::python::python_print_exception();     \
  }

#define SPROKIT_PYTHON_TRANSLATE_EXCEPTION(call)   \
  try                                              \
  {                                                \
    call;                                          \
  }                                                \
  catch (std::exception const& e)                  \
  {                                                \
    sprokit::python::python_gil const gil;         \
                                                   \
    (void)gil;                                     \
                                                   \
    PyErr_SetString(PyExc_RuntimeError, e.what()); \
                                                   \
    throw;                                         \
  }

SPROKIT_PYTHON_UTIL_EXPORT void python_print_exception();

}

}

#endif // SPROKIT_PYTHON_UTIL_PYTHON_EXCEPTIONS_H
