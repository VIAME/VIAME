// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_EXAMPLE_PROCESS_TRAITS_H
#define SPROKIT_EXAMPLE_PROCESS_TRAITS_H

#include <vital/vital_types.h>

create_type_trait( integer, "kwiver:test:integer", int32_t );

create_port_trait( integer, integer, "number uint 32" );

#endif // SPROKIT_EXAMPLE_PROCESS_TRAITS_H
