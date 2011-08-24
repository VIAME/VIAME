/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "basic_types.h"

/**
 * \file basic_types.cxx
 *
 * \brief Basic port types within the pipeline.
 */

namespace vistk
{

process::port_type_t const basic_types::t_bool = process::port_type_t("_bool");

process::port_type_t const basic_types::t_char = process::port_type_t("_char");
process::port_type_t const basic_types::t_string = process::port_type_t("_string");

process::port_type_t const basic_types::t_integer = process::port_type_t("_integer");
process::port_type_t const basic_types::t_unsigned = process::port_type_t("_unsigned");

process::port_type_t const basic_types::t_float = process::port_type_t("_float");
process::port_type_t const basic_types::t_double = process::port_type_t("_double");

process::port_type_t const basic_types::t_byte = process::port_type_t("_byte");

static process::port_type_t const vec_prefix = process::port_type_t("_vec");

process::port_type_t const basic_types::t_vec_char = vec_prefix + t_char;
process::port_type_t const basic_types::t_vec_string = vec_prefix + t_string;

process::port_type_t const basic_types::t_vec_integer = vec_prefix + t_integer;
process::port_type_t const basic_types::t_vec_unsigned = vec_prefix + t_unsigned;

process::port_type_t const basic_types::t_vec_float = vec_prefix + t_float;
process::port_type_t const basic_types::t_vec_double = vec_prefix + t_double;

process::port_type_t const basic_types::t_vec_byte = vec_prefix + t_byte;

}
