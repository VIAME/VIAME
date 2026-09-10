// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/track_set.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace kwiver {

namespace vital {

namespace python {

std::shared_ptr< kwiver::vital::track >
get_track( std::shared_ptr< kwiver::vital::track_set >& self, uint64_t id )
{
  auto track = self->get_track( id );
  if( !track )
  {
    throw py::index_error( "Track does not exist in set" );
  }
  return track;
}

} // namespace python

} // namespace vital

} // namespace kwiver

using namespace kwiver::vital::python;
PYBIND11_MODULE( track_set, m )
{
  py::class_< kwiver::vital::track_set,
    std::shared_ptr< kwiver::vital::track_set > >( m, "TrackSet" )
    .def( py::init<>() )
    .def(
      py::init< std::vector< std::shared_ptr< kwiver::vital::track > > >(),
      py::arg( "tracks" ) )
    .def( "all_frame_ids", &kwiver::vital::track_set::all_frame_ids )
    .def(
      "get_track", &get_track,
      py::arg( "id" ) )
    .def( "first_frame", &kwiver::vital::track_set::first_frame )
    .def( "last_frame", &kwiver::vital::track_set::last_frame )
    .def( "size", &kwiver::vital::track_set::size )
    .def( "tracks", &kwiver::vital::track_set::tracks )
    // viame's frame stabilizer builds its track set incrementally through
    // these two; they were bound before the rewrite and python-side track
    // bookkeeping has no substitute for them.
    .def(
      "insert",
      static_cast< void ( kwiver::vital::track_set::* )(
        kwiver::vital::track_sptr const& ) >(
        &kwiver::vital::track_set::insert ) )
    .def(
      "active_tracks", &kwiver::vital::track_set::active_tracks,
      py::arg( "offset" ) = -1 )
    .def( "__len__", &kwiver::vital::track_set::size )
  ;
}
