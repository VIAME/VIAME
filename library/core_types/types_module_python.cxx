// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file algorithm_implementation.cxx
 *
 * \brief python bindings for algorithm
 */

#include <pybind11/pybind11.h>
#include "image_python.h"
#include "image_container_python.h"

namespace py = pybind11;
using namespace kwiver::vital::python;

PYBIND11_MODULE( types, m )
{
  image
  ::image( m );
  image_container
  ::image_container( m );
}
