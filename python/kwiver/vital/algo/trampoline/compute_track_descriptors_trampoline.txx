// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file compute_track_descriptors_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::compute_track_descriptors \endlink
 */

#ifndef COMPUTE_TRACK_DESCRIPTORS_TRAMPOLINE_TXX
#define COMPUTE_TRACK_DESCRIPTORS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/compute_track_descriptors.h>

namespace kwiver {
namespace vital  {
namespace python {

template< class algorithm_def_ctd_base=
            kwiver::vital::algorithm_def<
               kwiver::vital::algo::compute_track_descriptors > >
class algorithm_def_ctd_trampoline :
      public algorithm_trampoline<algorithm_def_ctd_base>
{
  public:
    using algorithm_trampoline< algorithm_def_ctd_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        std::string,
        kwiver::vital::algorithm_def<
          kwiver::vital::algo::compute_track_descriptors>,
        type_name,
      );
    }
};

template< class compute_track_descriptors_base=
                  kwiver::vital::algo::compute_track_descriptors >
class compute_track_descriptors_trampoline :
      public algorithm_def_ctd_trampoline< compute_track_descriptors_base >
{
  public:
    using algorithm_def_ctd_trampoline< compute_track_descriptors_base >::
              algorithm_def_ctd_trampoline;

    kwiver::vital::track_descriptor_set_sptr
    compute( kwiver::vital::timestamp ts,
             kwiver::vital::image_container_sptr image_data,
             kwiver::vital::object_track_set_sptr tracks ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::track_descriptor_set_sptr,
        kwiver::vital::algo::compute_track_descriptors,
        compute,
        ts,
        image_data,
        tracks
      );
    }

    kwiver::vital::track_descriptor_set_sptr
      flush() override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::track_descriptor_set_sptr,
        kwiver::vital::algo::compute_track_descriptors,
        flush,
      );

    }
};
}
}
}
#endif
