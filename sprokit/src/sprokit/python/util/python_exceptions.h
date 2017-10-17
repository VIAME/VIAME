/*ckwg +29
 * Copyright 2012 by Kitware, Inc.
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

#ifndef SPROKIT_PYTHON_UTIL_PYTHON_EXCEPTIONS_H
#define SPROKIT_PYTHON_UTIL_PYTHON_EXCEPTIONS_H

#include "util-config.h"

namespace sprokit {
namespace python {

/// \todo More useful output?

#define SPROKIT_PYTHON_HANDLE_EXCEPTION(call)                     \
  try                                                             \
  {                                                               \
    call;                                                         \
  }                                                               \
  catch (pybind11::error_already_set const& e)                    \
  {                                                               \
    auto logger = kwiver::vital::get_logger("python_exceptions"); \
    LOG_WARN(logger, "Ignore Python Exception:\n" << e.what());   \
    sprokit::python::python_print_exception();                    \
                                                                  \
    throw;                                                        \
  }

#define SPROKIT_PYTHON_IGNORE_EXCEPTION(call)                     \
  try                                                             \
  {                                                               \
    call;                                                         \
  }                                                               \
  catch (pybind11::error_already_set const& e)                    \
  {                                                               \
    auto logger = kwiver::vital::get_logger("python_exceptions"); \
    LOG_WARN(logger, "Ignore Python Exception:\n" << e.what());   \
    sprokit::python::python_print_exception();                    \
  }

#define SPROKIT_PYTHON_TRANSLATE_EXCEPTION(call)                  \
  try                                                             \
  {                                                               \
    call;                                                         \
  }                                                               \
  catch (std::exception const& e)                                 \
  {                                                               \
    sprokit::python::python_gil const gil;                        \
                                                                  \
    (void)gil;                                                    \
                                                                  \
    PyErr_SetString(PyExc_RuntimeError, e.what());                \
                                                                  \
    throw;                                                        \
  }

SPROKIT_PYTHON_UTIL_EXPORT void python_print_exception();

}

}

#endif // SPROKIT_PYTHON_UTIL_PYTHON_EXCEPTIONS_H
