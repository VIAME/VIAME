/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "image_types.h"

#include "basic_types.h"

/**
 * \file image_types.h
 *
 * \brief Basic port types within the pipeline.
 */

namespace vistk
{

static process::port_type_t const image_suffix = process::port_type_t("_image");

process::port_type_t const image_types::t_bitmask = basic_types::t_bool + image_suffix;
process::port_type_t const image_types::t_bytemask = basic_types::t_byte + image_suffix;

static process::port_type_t const gray_type = process::port_type_t("_gray");

process::port_type_t const image_types::t_byte_grayscale = basic_types::t_byte + gray_type + image_suffix;
process::port_type_t const image_types::t_float_grayscale = basic_types::t_float + gray_type + image_suffix;

static process::port_type_t const rgb_type = process::port_type_t("_rgb");

process::port_type_t const image_types::t_byte_rgb = basic_types::t_byte + rgb_type + image_suffix;
process::port_type_t const image_types::t_float_rgb = basic_types::t_float + rgb_type + image_suffix;

process::port_type_t const image_types::t_ipl = process::port_type_t("_ipl") + image_suffix;
process::port_type_t const image_types::t_opencv = t_ipl;

}
