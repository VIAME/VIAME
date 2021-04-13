// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/metadata_map_io.h>
#include <python/kwiver/vital/algo/trampoline/metadata_map_io_trampoline.txx>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace kwiver {

namespace vital {

namespace python {

void
metadata_map_io( py::module& m )
{
  py::class_< kwiver::vital::algo::metadata_map_io,
              std::shared_ptr< kwiver::vital::algo::metadata_map_io >,
              kwiver::vital::algorithm_def< kwiver::vital::algo::metadata_map_io >,
              metadata_map_io_trampoline<> >( m, "MetadataMapIO" )
    .def( py::init() )
    .def_static( "static_type_name",
                 &kwiver::vital::algo::metadata_map_io::static_type_name )
    .def( "load",
          static_cast< kwiver::vital::metadata_map_sptr
                       (kwiver::vital::algo::metadata_map_io::*)
                       (std::string const&) const >
          ( &kwiver::vital::algo::metadata_map_io::load ) )
    .def( "load",
          static_cast< kwiver::vital::metadata_map_sptr
                       (kwiver::vital::algo::metadata_map_io::*)
                       ( std::istream&,
                         std::string const& ) const >
          ( &kwiver::vital::algo::metadata_map_io::load ) )
    .def( "save",
          static_cast< void(kwiver::vital::algo::metadata_map_io::*)
                       ( std::string const&,
                         kwiver::vital::metadata_map_sptr ) const >
          ( &kwiver::vital::algo::metadata_map_io::save ) )
    .def( "save",
          static_cast< void(kwiver::vital::algo::metadata_map_io::*)
                       ( std::ostream&, kwiver::vital::metadata_map_sptr,
                         std::string const& ) const >
          ( &kwiver::vital::algo::metadata_map_io::save ) );
}

} // namespace python

} // namespace vital

} // namespace kwiver
