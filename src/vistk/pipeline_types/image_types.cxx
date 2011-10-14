/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "image_types.h"

#include "basic_types.h"

/**
 * \file image_types.cxx
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

static process::port_type_t const bgr_type = process::port_type_t("_bgr");

process::port_type_t const image_types::t_byte_bgr = basic_types::t_byte + bgr_type + image_suffix;
process::port_type_t const image_types::t_float_bgr = basic_types::t_float + bgr_type + image_suffix;

static process::port_type_t const rgba_type = process::port_type_t("_rgba");

process::port_type_t const image_types::t_byte_rgba = basic_types::t_byte + rgba_type + image_suffix;
process::port_type_t const image_types::t_float_rgba = basic_types::t_float + rgba_type + image_suffix;

static process::port_type_t const bgra_type = process::port_type_t("_bgra");

process::port_type_t const image_types::t_byte_bgra = basic_types::t_byte + bgra_type + image_suffix;
process::port_type_t const image_types::t_float_bgra = basic_types::t_float + bgra_type + image_suffix;

static process::port_type_t const yuv_type = process::port_type_t("_yuv");

process::port_type_t const image_types::t_byte_yuv = basic_types::t_byte + yuv_type + image_suffix;
process::port_type_t const image_types::t_float_yuv = basic_types::t_float + yuv_type + image_suffix;

process::port_type_t const image_types::t_ipl = process::port_type_t("_ipl") + image_suffix;
process::port_type_t const image_types::t_opencv = t_ipl;

}
