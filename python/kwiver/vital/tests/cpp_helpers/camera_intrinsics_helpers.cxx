// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/camera_intrinsics.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

// Helpers to call pure virtual functions from base reference.
// We'll use these to test that these methods can be overriden in C++
PYBIND11_MODULE( camera_intrinsics_helpers, m )
{
  m.def( "clone", [] ( const kv::camera_intrinsics& self )
  {
    return self.clone();
  });

  m.def( "focal_length", [] ( const kv::camera_intrinsics& self )
  {
    return self.focal_length();
  });

  m.def( "principal_point", [] ( const kv::camera_intrinsics& self )
  {
    return self.principal_point();
  });

  m.def( "aspect_ratio", [] ( const kv::camera_intrinsics& self )
  {
    return self.aspect_ratio();
  });

  m.def( "skew", [] ( const kv::camera_intrinsics& self )
  {
    return self.skew();
  });

  m.def( "image_width", [] ( const kv::camera_intrinsics& self )
  {
    return self.image_width();
  });

  m.def( "image_height", [] ( const kv::camera_intrinsics& self )
  {
    return self.image_height();
  });

  m.def( "dist_coeffs", [] ( const kv::camera_intrinsics& self )
  {
    return self.dist_coeffs();
  });

  m.def( "as_matrix", [] ( const kv::camera_intrinsics& self )
  {
    return self.as_matrix();
  });

  m.def( "map", [] ( const kv::camera_intrinsics& self, const kv::vector_2d& norm_pt )
  {
    return self.map( norm_pt );
  });

  m.def( "map", [] ( const kv::camera_intrinsics& self, const kv::vector_3d& norm_hpt )
  {
    return self.map( norm_hpt );
  });

  m.def( "unmap", [] ( const kv::camera_intrinsics& self, const kv::vector_2d& norm_pt )
  {
    return self.unmap( norm_pt );
  });

  m.def( "distort", [] ( const kv::camera_intrinsics& self, const kv::vector_2d& norm_pt )
  {
    return self.distort( norm_pt );
  });

  m.def( "undistort", [] ( const kv::camera_intrinsics& self, const kv::vector_2d& dist_pt )
  {
    return self.undistort( dist_pt );
  });

  m.def( "is_map_valid", [] ( const kv::camera_intrinsics& self, const kv::vector_2d& norm_pt )
  {
    return self.is_map_valid( norm_pt );
  });

  m.def( "is_map_valid", [] ( const kv::camera_intrinsics& self, const kv::vector_3d& norm_hpt )
  {
    return self.is_map_valid( norm_hpt );
  });

}
