// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file transform_2d_io_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<transform_2d_io> and transform_2d_io
 */

#ifndef TRANSFORM_2D_IO_TRAMPOLINE_TXX
#define TRANSFORM_2D_IO_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/transform_2d_io.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_t2dio_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::transform_2d_io > >
class algorithm_def_t2dio_trampoline :
      public algorithm_trampoline<algorithm_def_t2dio_base>
{
  public:
    using algorithm_trampoline<algorithm_def_t2dio_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::transform_2d_io>,
        type_name,
      );
    }
};

template< class transform_2d_io_base=
                kwiver::vital::algo::transform_2d_io >
class transform_2d_io_trampoline :
      public algorithm_def_t2dio_trampoline< transform_2d_io_base >
{
  public:
    using algorithm_def_t2dio_trampoline< transform_2d_io_base>::
              algorithm_def_t2dio_trampoline;

    kwiver::vital::transform_2d_sptr
    load_( std::string const& filename )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::transform_2d_sptr,
        kwiver::vital::algo::transform_2d_io,
        load_,
        filename
      );
    }

    void
    save_( std::string const& filename,
           kwiver::vital::transform_2d_sptr data ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::transform_2d_io,
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
