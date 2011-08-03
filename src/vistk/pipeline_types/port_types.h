/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_TYPES_PORT_TYPES_H
#define VISTK_PIPELINE_TYPES_PORT_TYPES_H

#include "pipeline_types-config.h"

#include <vistk/pipeline/process.h>

namespace vistk
{

class VISTK_PIPELINE_TYPES_EXPORT port_types
{
  public:
    // Raw types

    // Readable data
    static process::port_type_name_t const t_char;
    static process::port_type_name_t const t_string;

    // Number data
    static process::port_type_name_t const t_integer;
    static process::port_type_name_t const t_unsigned;

    // Float data
    static process::port_type_name_t const t_float;
    static process::port_type_name_t const t_double;

    // Binary data
    static process::port_type_name_t const t_byte;

    // Vector types

    // Readable data
    static process::port_type_name_t const t_vec_char;
    static process::port_type_name_t const t_vec_string;

    // Number data
    static process::port_type_name_t const t_vec_integer;
    static process::port_type_name_t const t_vec_unsigned;

    // Float data
    static process::port_type_name_t const t_vec_float;
    static process::port_type_name_t const t_vec_double;

    // Binary data
    static process::port_type_name_t const t_vec_byte;
};

}

#endif // VISTK_PIPELINE_TYPES_PORT_TYPES_H
