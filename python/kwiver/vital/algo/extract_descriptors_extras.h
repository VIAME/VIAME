// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_PYTHON_VITAL_ALGO_EXTRACT_DESCRIPTORS_EXTRAS_H_
#define KWIVER_PYTHON_VITAL_ALGO_EXTRACT_DESCRIPTORS_EXTRAS_H_

#include <pybind11/pybind11.h>

namespace kwiver::vital::python {

// Hand-written adjustments layered over the generated ExtractDescriptors
// binding; must run after extract_descriptors(m).
void extract_descriptors_extras( pybind11::module& m );

} // namespace kwiver::vital::python

#endif
