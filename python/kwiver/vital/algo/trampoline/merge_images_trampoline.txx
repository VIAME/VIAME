// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file merge_images_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<merge_images> and merge_images
 */

#ifndef MERGE_IMAGES_TRAMPOLINE_TXX
#define MERGE_IMAGES_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/merge_images.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_mi_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::merge_images > >
class algorithm_def_mi_trampoline :
      public algorithm_trampoline<algorithm_def_mi_base>
{
  public:
    using algorithm_trampoline<algorithm_def_mi_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::merge_images>,
        type_name,
      );
    }
};

template< class merge_images_base=
                kwiver::vital::algo::merge_images >
class merge_images_trampoline :
      public algorithm_def_mi_trampoline< merge_images_base >
{
  public:
    using algorithm_def_mi_trampoline< merge_images_base>::
              algorithm_def_mi_trampoline;

    kwiver::vital::image_container_sptr
    merge( kwiver::vital::image_container_sptr image1,
           kwiver::vital::image_container_sptr image2 ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::image_container_sptr,
        kwiver::vital::algo::merge_images,
        merge,
        image1,
        image2
      );
    }
};

}
}
}

#endif
