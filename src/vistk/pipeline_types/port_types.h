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

/**
 * \class port_types port_types.h <vistk/pipeline_types/port_types.h>
 *
 * \brief Basic port types.
 */
class VISTK_PIPELINE_TYPES_EXPORT port_types
{
  public:
    // Raw types

    /// The type for character data on a port.
    static process::port_type_t const t_char;
    /// The type for string data on a port.
    static process::port_type_t const t_string;

    /// The type for integer data on a port.
    static process::port_type_t const t_integer;
    /// The type for unsigned integer data on a port.
    static process::port_type_t const t_unsigned;

    /// The type for floating point data on a port.
    static process::port_type_t const t_float;
    /// The type for floating point double precision data on a port.
    static process::port_type_t const t_double;

    /// The type for a raw byte on a port.
    static process::port_type_t const t_byte;

    // Vector types

    /// The type for a sequence of character data on a port.
    static process::port_type_t const t_vec_char;
    /// The type for a sequence of string data on a port.
    static process::port_type_t const t_vec_string;

    /// The type for a sequence of integer data on a port.
    static process::port_type_t const t_vec_integer;
    /// The type for a sequence of unsigned integer data on a port.
    static process::port_type_t const t_vec_unsigned;

    /// The type for a sequence of floating point data on a port.
    static process::port_type_t const t_vec_float;
    /// The type for a sequence of floating point double precision data on a port.
    static process::port_type_t const t_vec_double;

    /// The type for a sequence of raw bytes on a port.
    static process::port_type_t const t_vec_byte;
};

}

#endif // VISTK_PIPELINE_TYPES_PORT_TYPES_H
