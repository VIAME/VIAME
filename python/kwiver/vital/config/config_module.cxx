// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/config/config.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace kwiver::vital::python;

PYBIND11_MODULE(config, m)
{
  config(m);
}
