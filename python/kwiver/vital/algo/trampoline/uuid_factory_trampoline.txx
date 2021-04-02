// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file uuid_factory_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<uuid_factory> and uuid_factory
 */

#ifndef UUID_FACTORY_TRAMPOLINE_TXX
#define UUID_FACTORY_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/uuid_factory.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_uf_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::uuid_factory > >
class algorithm_def_uf_trampoline :
      public algorithm_trampoline<algorithm_def_uf_base>
{
  public:
    using algorithm_trampoline<algorithm_def_uf_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::uuid_factory>,
        type_name,
      );
    }
};

template< class uuid_factory_base=
                kwiver::vital::algo::uuid_factory >
class uuid_factory_trampoline :
      public algorithm_def_uf_trampoline< uuid_factory_base >
{
  public:
    using algorithm_def_uf_trampoline< uuid_factory_base>::
              algorithm_def_uf_trampoline;

    kwiver::vital::uid
    create_uuid( ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::uid,
        kwiver::vital::algo::uuid_factory,
        create_uuid,
      );
    }

};

}
}
}

#endif
