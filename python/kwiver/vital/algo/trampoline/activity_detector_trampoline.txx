// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file image_object_detector_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of algorithm_def<image_object_detector> and image_object_detector
 */

#ifndef ACTIVTY_DETECTOR_TRAMPOLINE_TXX
#define ACTIVTY_DETECTOR_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/activity_detector.h>
#include <vital/types/activity.h>
#include <vital/types/image_container.h>

namespace kwiver {
namespace vital  {
namespace python {
template <class algorithm_def_ad_base=kwiver::vital::algorithm_def<kwiver::vital::algo::activity_detector>>
class algorithm_def_ad_trampoline :
      public algorithm_trampoline<algorithm_def_ad_base>
{
  public:
    using algorithm_trampoline<algorithm_def_ad_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::activity_detector>,
        type_name,
      );
    }
};

template <class activity_detector_base=kwiver::vital::algo::activity_detector>
class activity_detector_trampoline :
      public algorithm_def_ad_trampoline<activity_detector_base>
{
  public:
    using algorithm_def_ad_trampoline<activity_detector_base>::
              algorithm_def_ad_trampoline;
    std::vector<kwiver::vital::activity> detect(kwiver::vital::image_container_sptr image_data) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        std::vector<kwiver::vital::activity>,
        kwiver::vital::algo::activity_detector,
        detect,
        image_data
      );
    }
};
}
}
}
#endif
