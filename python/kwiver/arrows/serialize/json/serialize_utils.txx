// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_PYTHON_ARROW_SERIALIZE_JSON_SERIALIZE_UTILS_H_
#define KWIVER_PYTHON_ARROW_SERIALIZE_JSON_SERIALIZE_UTILS_H_

#include <vital/any.h>
#include <utility>
#include <memory>

namespace kwiver {
namespace python {
namespace arrows {
namespace json {

  template < typename type, typename serializer >
  std::string serialize( type t )
  {
    serializer serializer_algo{};
    kwiver::vital::any any_t{ t };
    return *serializer_algo.serialize(any_t);
  }

  template < typename type, typename serializer >
  type deserialize( const std::string& message )
  {
    serializer serializer_algo{};
    kwiver::vital::any any_t{ serializer_algo.deserialize( message ) };
    return kwiver::vital::any_cast< type >( any_t);
  }

}
}
}
}

#endif
