// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file image_io_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<image_io> and image_io
 */

#ifndef IMAGE_IO_TRAMPOLINE_TXX
#define IMAGE_IO_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/image_io.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/image_container.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_iio_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::image_io > >
class algorithm_def_iio_trampoline :
      public algorithm_trampoline<algorithm_def_iio_base>
{
  public:
    using algorithm_trampoline<algorithm_def_iio_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::image_io>,
        type_name,
      );
    }
};

template< class image_io_base=kwiver::vital::algo::image_io >
class image_io_trampoline :
      public algorithm_def_iio_trampoline< image_io_base >
{
  public:
    using algorithm_def_iio_trampoline< image_io_base>::
              algorithm_def_iio_trampoline;

    kwiver::vital::image_container_sptr
      load_( std::string const& filename ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::image_container_sptr,
        kwiver::vital::algo::image_io,
        load_,
        filename
      );
    }

    kwiver::vital::metadata_sptr
      load_metadata_( std::string const& filename ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::metadata_sptr,
        kwiver::vital::algo::image_io,
        load_metadata_,
        filename
      );
    }

    void
      save_( std::string const& filename,
             kwiver::vital::image_container_sptr data ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::image_io,
        save_,
        filename,
        data
      );
    }
};

}
}
}

#endif
