// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_PYTHON_DETECTED_OBJECT_SET_OUTPUT_H_
#define KWIVER_VITAL_PYTHON_DETECTED_OBJECT_SET_OUTPUT_H_

#include <pybind11/pybind11.h>
namespace kwiver {
namespace vital  {
namespace python {
void detected_object_set_output(pybind11::module &m);
}
}
}
#endif
