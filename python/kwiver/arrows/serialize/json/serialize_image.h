// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_PYTHON_ARROW_SERIALIZE_JSON_SERIALIZE_IMAGE_H_
#define KWIVER_PYTHON_ARROW_SERIALIZE_JSON_SERIALIZE_IMAGE_H_

#include <vital/types/image_container.h>
#include <pybind11/pybind11.h>

namespace kwiver {
namespace arrows {
namespace python {
namespace py = pybind11;

std::string
serialize_image_using_json( kwiver::vital::simple_image_container img );

kwiver::vital::simple_image_container
deserialize_image_using_json( const std::string& message );

void serialize_image(py::module &m);
}
}
}
#endif
