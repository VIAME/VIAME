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
      kwiver::vital::python::gil_scoped_acquire gil;
      pybind11::function overload = pybind11::get_overload( static_cast< const kwiver::vital::algo::extract_descriptors* > ( this ), "extract" );
      if( overload )
      {
        auto o = overload( image_data, features, image_mask );
        auto r = o.cast< std::pair< kwiver::vital::descriptor_set_sptr, kwiver::vital::feature_set_sptr > >();
        features = std::move( r.second );
        return r.first;
      }
      pybind11::pybind11_fail( "Tried to call pure virtual function \"kwiver::vital::algo::extract_descriptors::extract\"" );
    }
};

}
}
}

#endif
