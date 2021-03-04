// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file detected_object_filter_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::detected_object_filter \endlink
 */

#ifndef DETECTED_OBJECT_FILTER_TRAMPOLINE_TXX
#define DETECTED_OBJECT_FILTER_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/detected_object_filter.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_dof_base=
           kwiver::vital::algorithm_def< kwiver::vital::algo::detected_object_filter > >
class algorithm_def_dof_trampoline :
      public algorithm_trampoline< algorithm_def_dof_base>
{
  public:
    using algorithm_trampoline< algorithm_def_dof_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::detected_object_filter>,
        type_name,
      );
    }
};

template< class detected_object_filter_base=kwiver::vital::algo::detected_object_filter >
class detected_object_filter_trampoline :
      public algorithm_def_dof_trampoline< detected_object_filter_base >
{
  public:
    using algorithm_def_dof_trampoline< detected_object_filter_base >::
              algorithm_def_dof_trampoline;

    kwiver::vital::detected_object_set_sptr
    filter( kwiver::vital::detected_object_set_sptr input_set ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::detected_object_set_sptr,
        kwiver::vital::algo::detected_object_filter,
        filter,
        input_set
      );
    }
};
}
}
}
#endif
