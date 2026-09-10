// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   cluster_info.cxx
 * @brief  Implementation for cluster_info class
 */

#include "cluster_info.h"

namespace sprokit {

cluster_info
::cluster_info( process::type_t const&         type_,
                process::description_t const&  description_,
                process_factory_func_t const&  ctor_ )
  : type( type_ ),
    description( description_ ),
    ctor( ctor_ )
{
}

} // end namespace sprokit
