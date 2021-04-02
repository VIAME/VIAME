// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file feature_descriptor_io_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::feature_descriptor_io \endlink
 */

#ifndef FEATURE_DESCRIPTOR_IO_TRAMPOLINE_TXX
#define FEATURE_DESCRIPTOR_IO_TRAMPOLINE_TXX

#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/feature_descriptor_io.h>

#include <python/kwiver/vital/util/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_fdio_base=
           kwiver::vital::algorithm_def<
               kwiver::vital::algo::feature_descriptor_io > >
class algorithm_def_fdio_trampoline :
      public algorithm_trampoline< algorithm_def_fdio_base >
{
  public:
    using algorithm_trampoline< algorithm_def_fdio_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def< kwiver::vital::algo::feature_descriptor_io >,
        type_name,
      );
    }
};

template< class feature_descriptor_io_base =
                  kwiver::vital::algo::feature_descriptor_io >
class feature_descriptor_io_trampoline :
      public algorithm_def_fdio_trampoline< feature_descriptor_io_base >
{
  public:
    using algorithm_def_fdio_trampoline< feature_descriptor_io_base >::
              algorithm_def_fdio_trampoline;

    void
    load_( std::string const& filename,
           kwiver::vital::feature_set_sptr& feat,
           kwiver::vital::descriptor_set_sptr& desc ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::feature_descriptor_io,
        load_,
        feat,
        desc
      );
    }

    void
    save_( std::string const& filename,
           kwiver::vital::feature_set_sptr feat,
           kwiver::vital::descriptor_set_sptr desc ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::feature_descriptor_io,
        save_,
        feat,
        desc
      );
    }
};

}
}
}

#endif
