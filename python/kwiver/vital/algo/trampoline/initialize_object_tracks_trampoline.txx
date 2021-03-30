// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file initialize_object_tracks_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<initialize_object_tracks> and initialize_object_tracks
 */

#ifndef INITIALIZE_OBJECT_TRACKS_TRAMPOLINE_TXX
#define INITIALIZE_OBJECT_TRACKS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/initialize_object_tracks.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_iot_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::initialize_object_tracks > >
class algorithm_def_iot_trampoline :
      public algorithm_trampoline<algorithm_def_iot_base>
{
  public:
    using algorithm_trampoline<algorithm_def_iot_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::initialize_object_tracks>,
        type_name,
      );
    }
};

template< class initialize_object_tracks_base=
                kwiver::vital::algo::initialize_object_tracks >
class initialize_object_tracks_trampoline :
      public algorithm_def_iot_trampoline< initialize_object_tracks_base >
{
  public:
    using algorithm_def_iot_trampoline< initialize_object_tracks_base>::
              algorithm_def_iot_trampoline;

    kwiver::vital::object_track_set_sptr
    initialize( kwiver::vital::timestamp ts,
                kwiver::vital::image_container_sptr image,
                kwiver::vital::detected_object_set_sptr detections ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::object_track_set_sptr,
        kwiver::vital::algo::initialize_object_tracks,
        initialize,
        ts,
        image,
        detections
      );
    }
};

}
}
}

#endif
