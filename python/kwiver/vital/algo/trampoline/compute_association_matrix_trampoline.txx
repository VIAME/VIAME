// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file compute_associate_matrix.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::compute_associate_matrix \endlink
 */

#ifndef COMPUTE_ASSOCAITION_MATRIX_TXX
#define COMPUTE_ASSOCIATION_MATRIX_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/compute_association_matrix.h>

namespace kwiver {
namespace vital  {
namespace python {

template< class algorithm_def_cam_base=
            kwiver::vital::algorithm_def<
               kwiver::vital::algo::compute_association_matrix > >
class algorithm_def_cam_trampoline :
      public algorithm_trampoline<algorithm_def_cam_base>
{
  public:
    using algorithm_trampoline< algorithm_def_cam_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
          kwiver::vital::algo::compute_association_matrix>,
        type_name,
      );
    }
};

template< class compute_association_matrix_base=
                  kwiver::vital::algo::compute_association_matrix >
class compute_association_matrix_trampoline :
      public algorithm_def_cam_trampoline< compute_association_matrix_base >
{
  public:
    using algorithm_def_cam_trampoline< compute_association_matrix_base >::
              algorithm_def_cam_trampoline;

    bool compute( kwiver::vital::timestamp ts,
                  kwiver::vital::image_container_sptr image,
                  kwiver::vital::object_track_set_sptr tracks,
                  kwiver::vital::detected_object_set_sptr detections,
                  kwiver::vital::matrix_d& matrix,
                  kwiver::vital::detected_object_set_sptr& considered ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algo::compute_association_matrix,
        compute,
        ts,
        image,
        tracks,
        detections,
        matrix,
        considered
      );
    }
};
}
}
}
#endif
