// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/metadata_map.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
namespace py = pybind11;
namespace kv = kwiver::vital;

// Helpers to call pure virtual functions from base reference.
// We'll use these to test that these methods can be overriden in C++
PYBIND11_MODULE( metadata_map_helpers, m )
{
  m.def( "size", [] ( const kv::metadata_map& self )
  {
    return self.size();
  });

  m.def( "metadata", [] ( const kv::metadata_map& self )
  {
    return self.metadata();
  });

  m.def( "has_item", [] ( const kv::metadata_map& self, kv::vital_metadata_tag tag,
                          kv::frame_id_t fid )
  {
    return self.has_item(tag, fid);
  });

  m.def( "get_vector", [] ( const kv::metadata_map& self, kv::frame_id_t fid )
  {
    return self.get_vector(fid);
  });

  m.def( "frames", [] ( kv::metadata_map& self )
  {
    return self.frames();
  });
}
