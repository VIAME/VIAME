// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file extract_descriptors_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::extract_descriptors \endlink
 */

#ifndef EXTRACT_DESCRIPTORS_TRAMPOLINE_TXX
#define EXTRACT_DESCRIPTORS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/extract_descriptors.h>

namespace kwiver {
namespace vital  {
namespace python {

template< class algorithm_def_ed_base=
           kwiver::vital::algorithm_def<
               kwiver::vital::algo::extract_descriptors > >
class algorithm_def_ed_trampoline :
      public algorithm_trampoline< algorithm_def_ed_base>
{
  public:
    using algorithm_trampoline< algorithm_def_ed_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def< kwiver::vital::algo::extract_descriptors >,
        type_name,
      );
    }
};

template< class extract_descriptors_base =
                  kwiver::vital::algo::extract_descriptors >
class extract_descriptors_trampoline :
      public algorithm_def_ed_trampoline< extract_descriptors_base >
{
  public:
    using algorithm_def_ed_trampoline< extract_descriptors_base >::
              algorithm_def_ed_trampoline;

    kwiver::vital::descriptor_set_sptr
    extract( kwiver::vital::image_container_sptr image_data,
             kwiver::vital::feature_set_sptr& features,
             kwiver::vital::image_container_sptr image_mask ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::descriptor_set_sptr,
        kwiver::vital::algo::extract_descriptors,
        extract,
        image_data,
        features,
        image_mask
      );
    }
};

}
}
}

#endif
