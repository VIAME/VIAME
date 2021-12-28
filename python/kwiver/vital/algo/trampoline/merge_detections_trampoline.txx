// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file merge_detections_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<merge_detections> and merge_detections
 */

#ifndef MERGE_DETECTIONS_TRAMPOLINE_TXX
#define MERGE_DETECTIONS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/merge_detections.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_md_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::merge_detections > >
class algorithm_def_md_trampoline :
      public algorithm_trampoline<algorithm_def_md_base>
{
  public:
    using algorithm_trampoline<algorithm_def_md_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::merge_detections>,
        type_name,
      );
    }
};

template< class merge_detections_base=
                kwiver::vital::algo::merge_detections >
class merge_detections_trampoline :
      public algorithm_def_md_trampoline< merge_detections_base >
{
  public:
    using algorithm_def_md_trampoline< merge_detections_base>::
              algorithm_def_md_trampoline;

    kwiver::vital::detected_object_set_sptr
    merge( std::vector<kwiver::vital::detected_object_set_sptr> const& sets ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::detected_object_set_sptr,
        kwiver::vital::algo::merge_detections,
        merge,
        sets
      );
    }
};

}
}
}

#endif
