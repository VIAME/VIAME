/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "port_types.h"

namespace vistk
{

namespace port_types
{

// Raw types

// Readable data
process::port_type_name_t const t_char = process::port_type_name_t("_char");
process::port_type_name_t const t_string = process::port_type_name_t("_string");

// Number data
process::port_type_name_t const t_integer = process::port_type_name_t("_integer");
process::port_type_name_t const t_unsigned = process::port_type_name_t("_unsigned");

// Float data
process::port_type_name_t const t_float = process::port_type_name_t("_float");
process::port_type_name_t const t_double = process::port_type_name_t("_double");

// Binary data
process::port_type_name_t const t_byte = process::port_type_name_t("_byte");

// Vector types

static process::port_type_name_t const vec_prefix = process::port_type_name_t("_vec");

// Readable data
process::port_type_name_t const t_vec_char = vec_prefix + t_char;
process::port_type_name_t const t_vec_string = vec_prefix + t_string;

// Number data
process::port_type_name_t const t_vec_integer = vec_prefix + t_integer;
process::port_type_name_t const t_vec_unsigned = vec_prefix + t_unsigned;

// Float data
process::port_type_name_t const t_vec_float = vec_prefix + t_float;
process::port_type_name_t const t_vec_double = vec_prefix + t_double;

// Binary data
process::port_type_name_t const t_vec_byte = vec_prefix + t_byte;

}

}
