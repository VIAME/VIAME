// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/track_set.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "python_fold.h"

namespace py = pybind11;

namespace viame {

namespace python {

std::shared_ptr< viame::track >
get_track( std::shared_ptr< viame::track_set >& self, uint64_t id )
{
  auto track = self->get_track( id );
  if( !track )
  {
    throw py::index_error( "Track does not exist in set" );
  }
  return track;
}

} // namespace python

} // namespace viame

using namespace viame::python;
VIAME_PYTHON_MODULE( track_set, m )
{
  py::class_< viame::track_set,
    std::shared_ptr< viame::track_set > >( m, "TrackSet" )
    .def( py::init<>() )
    .def(
      py::init< std::vector< std::shared_ptr< viame::track > > >(),
      py::arg( "tracks" ) )
    .def( "all_frame_ids", &viame::track_set::all_frame_ids )
    .def(
      "get_track", &get_track,
      py::arg( "id" ) )
    .def( "first_frame", &viame::track_set::first_frame )
    .def( "last_frame", &viame::track_set::last_frame )
    .def( "size", &viame::track_set::size )
    .def( "tracks", &viame::track_set::tracks )
    // viame's frame stabilizer builds its track set incrementally through
    // these two; they were bound before the rewrite and python-side track
    // bookkeeping has no substitute for them.
    .def(
      "insert",
      static_cast< void ( viame::track_set::* )(
        viame::track_sptr const& ) >(
        &viame::track_set::insert ) )
    .def(
      "active_tracks", &viame::track_set::active_tracks,
      py::arg( "offset" ) = -1 )
    .def( "__len__", &viame::track_set::size )
  ;
}
