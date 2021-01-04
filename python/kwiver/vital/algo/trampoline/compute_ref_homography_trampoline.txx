// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file compute_ref_homography.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::compute_ref_homography \endlink
 */

#ifndef COMPUTE_REF_HOMOGRAPHY_TXX
#define COMPUTE_REF_HOMOGRAPHY_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/compute_ref_homography.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_crh_base=
            kwiver::vital::algorithm_def<
               kwiver::vital::algo::compute_ref_homography > >
class algorithm_def_crh_trampoline :
      public algorithm_trampoline<algorithm_def_crh_base>
{
  public:
    using algorithm_trampoline< algorithm_def_crh_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
          kwiver::vital::algo::compute_ref_homography>,
        type_name,
      );
    }
};

template< class compute_ref_homography_base=
                  kwiver::vital::algo::compute_ref_homography >
class compute_ref_homography_trampoline :
      public algorithm_def_crh_trampoline< compute_ref_homography_base >
{
  public:
    using algorithm_def_crh_trampoline< compute_ref_homography_base >::
              algorithm_def_crh_trampoline;

    kwiver::vital::f2f_homography_sptr
    estimate( kwiver::vital::frame_id_t frame_number,
              kwiver::vital::feature_track_set_sptr tracks )
         const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::f2f_homography_sptr,
        kwiver::vital::algo::compute_ref_homography,
        estimate,
        frame_number,
        tracks
      );
    }
};
}
}
}
#endif
