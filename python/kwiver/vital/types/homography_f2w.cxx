// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/homography_f2w.h>

#include <pybind11/pybind11.h>

namespace py=pybind11;
namespace kv=kwiver::vital;

PYBIND11_MODULE( homography_f2w, m )
{
  py::class_< kv::f2w_homography, std::shared_ptr< kv::f2w_homography > >( m, "F2WHomography" )
  .def( py::init< kv::frame_id_t const >() )
  .def( py::init< kv::homography_sptr const&, kv::frame_id_t const >() )
  .def( py::init< kv::f2w_homography const& >() )
  .def_property_readonly( "homography", &kv::f2w_homography::homography )
  .def_property_readonly( "frame_id", &kv::f2w_homography::frame_id )
  .def( "get",
   [] ( kv::f2w_homography const& self, int r, int c )
   {
     auto m = self.homography()->matrix();
     if( 0 <= r && r < m.rows() && 0 <= c && c < m.cols() )
     {
       return m( r, c );
     }
     throw std::out_of_range( "Tried to perform get() out of bounds" );
   },
   "Convenience method that returns the underlying coefficient"
   " at the given row and column" )
  ;
}
