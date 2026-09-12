// This file is part of VIAME, and is distributed under an OSI-approved
// BSD 3-Clause License. See either the root top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

#ifndef KWIVER_PYTHON_ALGO_ESTIMATOR_EXTRAS_H
#define KWIVER_PYTHON_ALGO_ESTIMATOR_EXTRAS_H

#include <pybind11/pybind11.h>

namespace kwiver::vital::python {

void estimator_extras( pybind11::module& m );

void optimize_cameras_extras( pybind11::module& m );

} // namespace kwiver::vital::python

#endif
