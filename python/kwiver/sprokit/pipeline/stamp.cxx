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

#include <pybind11/pybind11.h>

#include "python_wrappers.cxx"

/**
 * \file stamp.cxx
 *
 * \brief Python bindings for \link sprokit::stamp\endlink.
 */

using namespace pybind11;

PYBIND11_MODULE(stamp, m)
{
  m.def("new_stamp", &new_stamp
    , (arg("increment"))
    , "Creates a new stamp.");
  m.def("incremented_stamp", &incremented_stamp
    , (arg("stamp"))
    , "Creates a stamp that is greater than the given stamp.");

  class_<wrap_stamp>(m, "Stamp"
    , "An identifier to help synchronize data within the pipeline.")
    .def("__eq__", &wrap_stamp::stamp_eq)
    .def("__lt__", &wrap_stamp::stamp_lt)
  ;

  // Equivalent to:
  //   @total_ordering
  //   class Stamp(object):
  //       ...
// XXX(python): 2.7
#if PY_VERSION_HEX >= 0x02070000
  object const functools = module::import("functools");
  object const total_ordering = functools.attr("total_ordering");
  #ifndef WIN32
    object const stamp = m.attr("Stamp");
    m.attr("Stamp") = total_ordering(stamp);
  #endif
#endif
}
