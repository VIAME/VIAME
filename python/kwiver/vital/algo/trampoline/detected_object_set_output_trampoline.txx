// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file detected_object_set_output_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::detected_object_set_output \endlink
 */

#ifndef DETECTED_OBJECT_SET_OUTPUT_TRAMPOLINE_TXX
#define DETECTED_OBJECT_SET_OUTPUT_TRAMPOLINE_TXX

#include <vital/algo/detected_object_set_output.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/image_container.h>
#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_doso_base=
           kwiver::vital::algorithm_def< kwiver::vital::algo::detected_object_set_output > >
class algorithm_def_doso_trampoline :
      public algorithm_trampoline< algorithm_def_doso_base>
{
  public:
    using algorithm_trampoline< algorithm_def_doso_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::detected_object_set_output>,
        type_name,
      );
    }
};

template< class detected_object_set_output_base=kwiver::vital::algo::detected_object_set_output >
class detected_object_set_output_trampoline :
      public algorithm_def_doso_trampoline< detected_object_set_output_base >
{
  public:
    using algorithm_def_doso_trampoline< detected_object_set_output_base >::
              algorithm_def_doso_trampoline;

    void
    open( std::string const& filename ) override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::detected_object_set_output,
        open,
        filename
      );
    }

    void
    close() override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::detected_object_set_output,
        close,
      );
    }

    void complete() override
    {
      VITAL_PYBIND11_OVERLOAD(
          void,
          kwiver::vital::algo::detected_object_set_output,
          complete,
          );
    }

    void
    write_set( kwiver::vital::detected_object_set_sptr const set,
               std::string const& image_name ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::detected_object_set_output,
        write_set,
        set,
        image_name
      );
    }
};
}
}
}
#endif
