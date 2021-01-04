// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>

#include <sprokit/pipeline/utils.h>

/**
 * \file utils.cxx
 *
 * \brief Python bindings for utils.
 */

using namespace pybind11;

PYBIND11_MODULE(utils, m)
{
  m.def("name_thread", &sprokit::name_thread
    , (arg("name"))
    , "Names the currently running thread.");
}
