// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>

#include "python_wrappers.cxx"

/**
 * \file stamp.cxx
 *
 * \brief Python bindings for \link sprokit::stamp\endlink.
 */

using namespace pybind11;
using namespace kwiver::sprokit::python;
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
