// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_PYTHON_METADATA_MAP_IO_H_
#define KWIVER_VITAL_PYTHON_METADATA_MAP_IO_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace kwiver {

namespace vital {

namespace python {

void metadata_map_io( py::module& m );

} // namespace python

} // namespace vital

} // namespace kwiver

#endif
