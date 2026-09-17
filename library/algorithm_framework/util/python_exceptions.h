// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_PYTHON_UTIL_PYTHON_EXCEPTIONS_H
#define VITAL_PYTHON_UTIL_PYTHON_EXCEPTIONS_H

#include <pybind11/pybind11.h>
#include <viame/algorithm_framework/util/viame_python_util_export.h>

namespace viame {

namespace python {

/// \todo More useful output?

void VIAME_PYTHON_UTIL_EXPORT python_print_exception();

#define VITAL_PYTHON_HANDLE_EXCEPTION( call )                     \
try                                                               \
{                                                                 \
  call;                                                           \
}                                                                 \
catch( pybind11::error_already_set const& e )                     \
{                                                                 \
  auto logger = viame::get_logger( "python_exceptions" ); \
  LOG_WARN( logger, "Ignore Python Exception:\n" << e.what() );   \
  viame::python::python_print_exception();                \
                                                                  \
  throw;                                                          \
}

#define VITAL_PYTHON_IGNORE_EXCEPTION( call )                     \
try                                                               \
{                                                                 \
  call;                                                           \
}                                                                 \
catch( pybind11::error_already_set const& e )                     \
{                                                                 \
  auto logger = viame::get_logger( "python_exceptions" ); \
  LOG_WARN( logger, "Ignore Python Exception:\n" << e.what() );   \
  viame::python::python_print_exception();                \
}

#define VITAL_PYTHON_TRANSLATE_EXCEPTION( call )   \
try                                                \
{                                                  \
  call;                                            \
}                                                  \
catch( std::exception const& e )                   \
{                                                  \
  pybind11::gil_scoped_acquire acquire;            \
  ( void ) acquire;                                \
  PyErr_SetString( PyExc_RuntimeError, e.what() ); \
                                                   \
  throw;                                           \
}

} // namespace python

} // namespace viame

#endif // VITAL_PYTHON_UTIL_PYTHON_EXCEPTIONS_H
