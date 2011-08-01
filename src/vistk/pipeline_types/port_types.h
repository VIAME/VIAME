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

namespace port_types
{

// Raw types

// Readable data
extern process::port_type_name_t const t_char;
extern process::port_type_name_t const t_string;

// Number data
extern process::port_type_name_t const t_integer;
extern process::port_type_name_t const t_unsigned;

// Float data
extern process::port_type_name_t const t_float;
extern process::port_type_name_t const t_double;

// Binary data
extern process::port_type_name_t const t_byte;

// Vector types

// Readable data
extern process::port_type_name_t const t_vec_char;
extern process::port_type_name_t const t_vec_string;

// Number data
extern process::port_type_name_t const t_vec_integer;
extern process::port_type_name_t const t_vec_unsigned;

// Float data
extern process::port_type_name_t const t_vec_float;
extern process::port_type_name_t const t_vec_double;

// Binary data
extern process::port_type_name_t const t_vec_byte;

}

}

#endif // VISTK_PIPELINE_TYPES_PORT_TYPES_H
