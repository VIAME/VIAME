// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file track_features_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<track_features> and track_features
 */

#ifndef TRACK_FEATURES_TRAMPOLINE_TXX
#define TRACK_FEATURES_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/track_features.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_tf_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::track_features > >
class algorithm_def_tf_trampoline :
      public algorithm_trampoline<algorithm_def_tf_base>
{
  public:
    using algorithm_trampoline<algorithm_def_tf_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::track_features>,
        type_name,
      );
    }
};

template< class track_features_base=
                kwiver::vital::algo::track_features >
class track_features_trampoline :
      public algorithm_def_tf_trampoline< track_features_base >
{
  public:
    using algorithm_def_tf_trampoline< track_features_base>::
              algorithm_def_tf_trampoline;

    kwiver::vital::feature_track_set_sptr
    track( kwiver::vital::feature_track_set_sptr prev_tracks,
           kwiver::vital::frame_id_t frame_number,
           kwiver::vital::image_container_sptr image_data,
           kwiver::vital::image_container_sptr mask ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::feature_track_set_sptr,
        kwiver::vital::algo::track_features,
        track,
        prev_tracks,
        frame_number,
        image_data,
        mask
      );
    }
};

}
}
}

#endif
