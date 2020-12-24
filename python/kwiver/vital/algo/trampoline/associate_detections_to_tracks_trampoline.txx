// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file associate_detections_to_tracks_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::associate_detections_to_tracks \endlink
 */

#ifndef ASSOCIATE_DETECTIONS_TO_TRACKS_TRAMPOLINE_TXX
#define ASSOCIATE_DETECTIONS_TO_TRACKS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/associate_detections_to_tracks.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_adtt_base=
            kwiver::vital::algorithm_def<
               kwiver::vital::algo::associate_detections_to_tracks > >
class algorithm_def_adtt_trampoline :
      public algorithm_trampoline<algorithm_def_adtt_base>
{
  public:
    using algorithm_trampoline< algorithm_def_adtt_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
          kwiver::vital::algo::associate_detections_to_tracks>,
        type_name,
      );
    }
};

template< class associate_detections_to_tracks_base=
                  kwiver::vital::algo::associate_detections_to_tracks >
class associate_detections_to_tracks_trampoline :
      public algorithm_def_adtt_trampoline< associate_detections_to_tracks_base >
{
  public:
    using algorithm_def_adtt_trampoline< associate_detections_to_tracks_base >::
              algorithm_def_adtt_trampoline;

    bool associate( kwiver::vital::timestamp ts,
                    kwiver::vital::image_container_sptr image,
                    kwiver::vital::object_track_set_sptr tracks,
                    kwiver::vital::detected_object_set_sptr detections,
                    kwiver::vital::matrix_d matrix,
                    kwiver::vital::object_track_set_sptr& output,
                    kwiver::vital::detected_object_set_sptr& unused ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algo::associate_detections_to_tracks,
        associate,
        ts,
        image,
        tracks,
        detections,
        matrix,
        output,
        unused
      );
    }
};
}
}
}
#endif
