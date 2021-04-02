// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file train_detector_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<train_detector> and train_detector
 */

#ifndef TRAIN_DETECTOR_TRAMPOLINE_TXX
#define TRAIN_DETECTOR_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/train_detector.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_td_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::train_detector > >
class algorithm_def_td_trampoline :
      public algorithm_trampoline<algorithm_def_td_base>
{
  public:
    using algorithm_trampoline<algorithm_def_td_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::train_detector>,
        type_name,
      );
    }
};

template< class train_detector_base=
                kwiver::vital::algo::train_detector >
class train_detector_trampoline :
      public algorithm_def_td_trampoline< train_detector_base >
{
  public:
    using algorithm_def_td_trampoline< train_detector_base>::
              algorithm_def_td_trampoline;

    void
    train_from_disk(
           kwiver::vital::category_hierarchy_sptr object_labels,
           std::vector< std::string > train_image_names,
           std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
           std::vector< std::string > test_image_names,
           std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth
         )  override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::train_detector,
        train_from_disk,
        object_labels,
        train_image_names,
        train_groundtruth,
        test_image_names,
        test_groundtruth
      );
    }

    void
    train_from_memory( kwiver::vital::category_hierarchy_sptr object_labels,
           std::vector< kwiver::vital::image_container_sptr > train_images,
           std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
           std::vector< kwiver::vital::image_container_sptr > test_images,
           std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth
         )  override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::train_detector,
        train_from_memory,
        object_labels,
        train_images,
        train_groundtruth,
        test_images,
        test_groundtruth
      );
    }
};

}
}
}

#endif
