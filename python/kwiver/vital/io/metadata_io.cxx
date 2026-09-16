// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/algorithm_framework/io/metadata_io.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

namespace py = pybind11;

// ----------------------------------------------------------------------------
PYBIND11_MODULE( metadata_io, m )
{
  m.def(
    "basename_from_metadata",
    ( std::string ( * )(
      viame::metadata_sptr,
      viame::frame_id_t ) ) & viame::basename_from_metadata );
  m.def(
    "basename_from_metadata",
    ( std::string ( * )(
      viame::metadata_vector const&,
      viame::frame_id_t ) ) & viame::basename_from_metadata );
}
