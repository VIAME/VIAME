// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_PYTHON_UTIL_PYTHON_EXCEPTIONS_H
#define VITAL_PYTHON_UTIL_PYTHON_EXCEPTIONS_H

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/util/vital_python_util_export.h>

namespace kwiver {
namespace vital  {
namespace python {

/// \todo More useful output?

void VITAL_PYTHON_UTIL_EXPORT python_print_exception();

#define VITAL_PYTHON_HANDLE_EXCEPTION(call)                     \
  try                                                             \
  {                                                               \
    call;                                                         \
  }                                                               \
  catch (pybind11::error_already_set const& e)                    \
  {                                                               \
    auto logger = kwiver::vital::get_logger("python_exceptions"); \
    LOG_WARN(logger, "Ignore Python Exception:\n" << e.what());   \
    kwiver::vital::python::python_print_exception();              \
                                                                  \
    throw;                                                        \
  }

#define VITAL_PYTHON_IGNORE_EXCEPTION(call)                     \
  try                                                             \
  {                                                               \
    call;                                                         \
  }                                                               \
  catch (pybind11::error_already_set const& e)                    \
  {                                                               \
    auto logger = kwiver::vital::get_logger("python_exceptions"); \
    LOG_WARN(logger, "Ignore Python Exception:\n" << e.what());   \
    kwiver::vital::python::python_print_exception();               \
  }

#define VITAL_PYTHON_TRANSLATE_EXCEPTION(call)                  \
  try                                                             \
  {                                                               \
    call;                                                         \
  }                                                               \
  catch (std::exception const& e)                                 \
  {                                                               \
    kwiver::vital::python::gil_scoped_acquire acquire;            \
    (void)acquire;                                                \
    PyErr_SetString(PyExc_RuntimeError, e.what());                \
                                                                  \
    throw;                                                        \
  }

} } }

#endif // VITAL_PYTHON_UTIL_PYTHON_EXCEPTIONS_H
