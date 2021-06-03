// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file refine_detections_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<refine_detections> and refine_detections
 */

#ifndef REFINE_DETECTIONS_TRAMPOLINE_TXX
#define REFINE_DETECTIONS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/refine_detections.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_rd_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::refine_detections > >
class algorithm_def_rd_trampoline :
      public algorithm_trampoline<algorithm_def_rd_base>
{
  public:
    using algorithm_trampoline<algorithm_def_rd_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::refine_detections>,
        type_name,
      );
    }
};

template< class refine_detections_base=
                kwiver::vital::algo::refine_detections >
class refine_detections_trampoline :
      public algorithm_def_rd_trampoline< refine_detections_base >
{
  public:
    using algorithm_def_rd_trampoline< refine_detections_base>::
              algorithm_def_rd_trampoline;

    kwiver::vital::detected_object_set_sptr
    refine( kwiver::vital::image_container_sptr image_data,
            kwiver::vital::detected_object_set_sptr detections ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::detected_object_set_sptr,
        kwiver::vital::algo::refine_detections,
        refine,
        image_data,
        detections
      );
    }
};

}
}
}

#endif
